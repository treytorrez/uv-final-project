#!/usr/bin/env python3
"""Final training script for CASAS HAR with sampled loading and XGBoost tuning.

This script is designed for constrained local machines:
- Loads CASAS data lazily with Polars scan + .sample() before collect.
- Applies label cleanup for odd prefixes like `r1.Sleep` / `r2.Dress`.
- Adds a general activity column from the mapping used in exploratory analysis.
- Supports two training APIs:
  1) scikit-learn wrapper (`xgboost.XGBClassifier`)
  2) native XGBoost learning API (`xgboost.train`)
- Tunes hyperparameters efficiently in both branches.
- Uses 5-fold K-Fold CV and macro-F1 as the primary model metric.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.preprocessing import LabelEncoder

# Sampling default requested by user.
DEFAULT_SAMPLE_FRACTION: float = 0.3
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_N_FOLDS: int = 5
DEFAULT_MAX_DEPTH: int = 5
DEFAULT_LEARNING_RATE: float = 0.05  # Slightly lower than common 0.1 default.
DEFAULT_N_TRIALS: int = 12
DEFAULT_RESERVE_THREADS: int = 4

DATA_GLOB: str = "data/raw/csh*/csh*.ann.features.csv"
MODELS_DIR: Path = Path("models")

GENERAL_ACTIVITY_MAP: dict[str, str] = {
    "Bathe": "Hygiene",
    "Personal_Hygiene": "Hygiene",
    "Groom": "Hygiene",
    "Toilet": "Hygiene",
    "Bed_Toilet_Transition": "Hygiene",
    "Dress": "Hygiene",
    "Cook": "Meal_Prep",
    "Cook_Breakfast": "Meal_Prep",
    "Cook_Lunch": "Meal_Prep",
    "Cook_Dinner": "Meal_Prep",
    "Wash_Dishes": "Meal_Cleanup",
    "Wash_Breakfast_Dishes": "Meal_Cleanup",
    "Wash_Lunch_Dishes": "Meal_Cleanup",
    "Wash_Dinner_Dishes": "Meal_Cleanup",
    "Eat": "Eating",
    "Drink": "Eating",
    "Eat_Breakfast": "Eating",
    "Eat_Lunch": "Eating",
    "Eat_Dinner": "Eating",
    "Sleep": "Sleep_Rest",
    "Go_To_Sleep": "Sleep_Rest",
    "Wake_Up": "Sleep_Rest",
    "Nap": "Sleep_Rest",
    "Sleep_Out_Of_Bed": "Sleep_Rest",
    "Work": "Work_Study",
    "Work_On_Computer": "Work_Study",
    "Work_At_Desk": "Work_Study",
    "Work_At_Table": "Work_Study",
    "Exercise": "Exercise",
    "Read": "Leisure",
    "Phone": "Leisure",
    "Relax": "Leisure",
    "Watch_TV": "Leisure",
    "Entertain_Guests": "Leisure",
    "Morning_Meds": "Medication",
    "Evening_Meds": "Medication",
    "Take_Medicine": "Medication",
    "Enter_Home": "Home_Transition",
    "Leave_Home": "Home_Transition",
    "Step_Out": "Home_Transition",
    "Laundry": "Household",
}


def compute_training_threads(reserve_threads: int = DEFAULT_RESERVE_THREADS) -> int:
    """Return the number of threads for training while leaving some cores free."""
    cpu_total = os.cpu_count() or 8
    return max(1, cpu_total - reserve_threads)


def load_sampled_dataset(sample_fraction: float, seed: int) -> pl.DataFrame:
    """Load CASAS data lazily and sample each home before global concatenation.

    This avoids collecting all homes at once. The current Polars version does not
    expose LazyFrame.sample(), so we collect per-home and immediately sample.
    """
    data_files = sorted(Path("data/raw").glob("csh*/csh*.ann.features.csv"))
    if not data_files:
        msg = f"No data files found for pattern: {DATA_GLOB}"
        raise FileNotFoundError(msg)

    sampled_frames: list[pl.DataFrame] = []
    for idx, file_path in enumerate(data_files):
        home_id = file_path.parent.name
        home_df = (
            pl.scan_csv(str(file_path))
            .with_columns(pl.lit(home_id).alias("home_id"))
            .with_columns(
                # r1./r2. cleanup from exploratory notebook.
                pl.col("activity").str.replace(r"^r[12]\.", "").alias("activity")
            )
            .filter(pl.col("activity") != "Other_Activity")
            .with_columns(
                pl.col("activity")
                .replace(GENERAL_ACTIVITY_MAP)
                .alias("activity_general")
            )
            .collect()
            .sample(fraction=sample_fraction, seed=seed + idx, shuffle=True)
        )
        sampled_frames.append(home_df)

    if not sampled_frames:
        msg = "Sampling produced zero frames."
        raise RuntimeError(msg)

    return pl.concat(sampled_frames, how="vertical")


def select_features(
    df: pl.DataFrame, target_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """Build numeric feature matrix X, raw target labels, and encoded target labels."""
    excluded = {"activity", "activity_general", "home_id"}
    feature_columns = [
        col
        for col in df.columns
        if col not in excluded and df[col].dtype.is_numeric()
    ]
    if not feature_columns:
        msg = "No numeric feature columns found."
        raise ValueError(msg)

    X = df.select(feature_columns).to_numpy().astype(np.float32)
    y_raw = df.get_column(target_column).to_numpy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    return X, y_raw, y_encoded, feature_columns, encoder


class LabelMappedXGBClassifier:
    """XGBClassifier wrapper that returns original labels on predict()."""

    def __init__(self, model: xgb.XGBClassifier, classes_: np.ndarray) -> None:
        self.model = model
        self.classes_ = np.array(classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_int = self.model.predict(X).astype(int)
        return self.classes_[pred_int]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def train_sklearn_api(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_trials: int,
    seed: int,
    train_threads: int,
    verbosity: int,
    class_labels: np.ndarray,
) -> tuple[LabelMappedXGBClassifier, float, dict[str, Any]]:
    """Train/tune with K-Fold + manual random search using XGBClassifier.

    XGBoost's sklearn wrapper requires fold-local contiguous class ids. With
    plain K-Fold, some classes can be absent in a training fold, so we remap
    y_train labels per fold before fitting.
    """
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    param_distributions: dict[str, list[Any]] = {
        "learning_rate": [0.02, 0.03, 0.05, 0.07],
        "max_depth": [4, 5, 6],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.05, 0.1],
        "reg_lambda": [0.8, 1.0, 1.2],
        "n_estimators": [300, 500, 800],
    }
    candidates = list(
        ParameterSampler(param_distributions=param_distributions, n_iter=n_trials, random_state=seed)
    )

    best_score = -1.0
    best_params: dict[str, Any] = {}
    all_trial_scores: list[dict[str, Any]] = []

    for trial_idx, params in enumerate(candidates, start=1):
        fold_scores: list[float] = []
        print(f"[sklearn-search] trial={trial_idx:02d}/{len(candidates)} params={params}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
            y_train_raw = y[train_idx]
            y_val_raw = y[val_idx]

            train_classes = np.unique(y_train_raw)
            train_class_to_int = {label: i for i, label in enumerate(train_classes)}
            y_train_fold = np.array(
                [train_class_to_int[label] for label in y_train_raw], dtype=np.int32
            )

            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=len(train_classes),
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=seed + trial_idx + fold_idx,
                n_jobs=train_threads,
                verbosity=verbosity,
                **params,
            )
            model.fit(X[train_idx], y_train_fold, verbose=False)

            pred_int = model.predict(X[val_idx]).astype(int)
            pred_raw = train_classes[pred_int]
            fold_score = float(
                f1_score(
                    y_val_raw,
                    pred_raw,
                    average="macro",
                    labels=class_labels.tolist(),
                    zero_division=0,
                )
            )
            fold_scores.append(fold_score)
            print(
                f"[sklearn-search] trial={trial_idx:02d} fold={fold_idx}/{n_folds} "
                f"macro_f1={fold_score:.5f}"
            )

        trial_score = float(np.mean(fold_scores))
        all_trial_scores.append(
            {
                "trial": trial_idx,
                "cv_macro_f1_mean": trial_score,
                "params": params,
            }
        )
        print(
            f"[sklearn-search] trial={trial_idx:02d} "
            f"macro_f1_mean={trial_score:.5f}"
        )

        if trial_score > best_score:
            best_score = trial_score
            best_params = params

    if not best_params:
        msg = "No valid sklearn parameter set was found during search."
        raise RuntimeError(msg)

    # Refit on full data with stable global class mapping.
    global_classes = np.array(class_labels)
    global_class_to_int = {label: i for i, label in enumerate(global_classes)}
    y_full = np.array([global_class_to_int[label] for label in y], dtype=np.int32)

    final_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(global_classes),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=train_threads,
        verbosity=verbosity,
        **best_params,
    )
    final_model.fit(X, y_full, verbose=False)

    details = {
        "best_params": best_params,
        "best_cv_f1_macro": best_score,
        "trial_scores": all_trial_scores,
    }
    return LabelMappedXGBClassifier(model=final_model, classes_=global_classes), best_score, details


def macro_f1_eval(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Custom metric for native XGBoost CV: macro-F1 from class probabilities."""
    y_true = dtrain.get_label().astype(int)
    n_classes = int(preds.size / max(1, y_true.shape[0]))
    proba = preds.reshape(-1, n_classes)
    pred_labels = np.argmax(proba, axis=1)
    score = f1_score(
        y_true,
        pred_labels,
        average="macro",
        labels=list(range(n_classes)),
        zero_division=0,
    )
    return "macro_f1", float(score)


def train_native_api(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_trials: int,
    seed: int,
    train_threads: int,
    verbosity: int,
) -> tuple[xgb.Booster, float, dict[str, Any]]:
    """Train/tune with native xgb.train and xgb.cv."""
    dtrain = xgb.DMatrix(X, label=y)
    n_classes = int(np.max(y)) + 1
    rng = np.random.default_rng(seed)
    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))

    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_rounds = 0

    for trial in range(1, n_trials + 1):
        params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.07])),
            "max_depth": int(rng.choice([4, 5, 6])),
            "min_child_weight": int(rng.choice([1, 3, 5])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
            "alpha": float(rng.choice([0.0, 0.05, 0.1])),
            "lambda": float(rng.choice([0.8, 1.0, 1.2])),
            "eval_metric": "mlogloss",
            "seed": seed + trial,
            "nthread": train_threads,
            "verbosity": verbosity,
        }

        cv_result = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=800,
            folds=folds,
            metrics=("mlogloss",),
            custom_metric=macro_f1_eval,
            maximize=True,
            early_stopping_rounds=40,
            seed=seed,
            verbose_eval=100 if verbosity >= 2 else False,
        )

        score_series = cv_result["test-macro_f1-mean"]
        score = float(score_series.max())
        rounds = int(score_series.idxmax() + 1)
        print(f"[native-search] trial={trial:02d} macro_f1={score:.5f} rounds={rounds}")

        if score > best_score:
            best_score = score
            best_params = params
            best_rounds = rounds

    if best_rounds <= 0:
        msg = "Native search failed to find valid boosting rounds."
        raise RuntimeError(msg)

    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=best_rounds,
        evals=[(dtrain, "train")],
        verbose_eval=100 if verbosity >= 2 else False,
    )

    details = {
        "best_params": best_params,
        "best_cv_f1_macro": best_score,
        "best_num_boost_round": best_rounds,
    }
    return model, best_score, details


def compute_train_metrics(
    model: xgb.XGBClassifier | LabelMappedXGBClassifier | xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute quick in-sample diagnostics for saved metadata."""
    if isinstance(model, xgb.Booster):
        pred_proba = model.predict(xgb.DMatrix(X))
        y_pred = np.argmax(pred_proba, axis=1)
    else:
        y_pred = model.predict(X)

    return {
        "train_f1_macro": float(f1_score(y, y_pred, average="macro")),
        "train_f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        "train_accuracy": float(accuracy_score(y, y_pred)),
    }


def save_outputs(
    model: xgb.XGBClassifier | LabelMappedXGBClassifier | xgb.Booster,
    details: dict[str, Any],
    metric_value: float,
    metric_name: str,
    api: str,
    target_column: str,
    feature_columns: list[str],
    label_encoder: LabelEncoder,
    sample_fraction: float,
    train_threads: int,
    train_metrics: dict[str, float],
) -> tuple[Path, Path]:
    """Save trained model and metadata with metric + timestamp in filename."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_slug = metric_name.replace(" ", "_")
    metric_fragment = f"{metric_slug}_{metric_value:.4f}"

    if api == "native":
        model_path = MODELS_DIR / f"xgb_native_{target_column}_{metric_fragment}_{ts}.json"
        model.save_model(model_path)
    else:
        model_path = MODELS_DIR / f"xgb_sklearn_{target_column}_{metric_fragment}_{ts}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

    meta_path = MODELS_DIR / f"meta_{api}_{target_column}_{metric_fragment}_{ts}.json"
    meta = {
        "timestamp": ts,
        "api": api,
        "target_column": target_column,
        "sample_fraction": sample_fraction,
        "train_threads": train_threads,
        "metric_primary": metric_name,
        "metric_primary_cv": metric_value,
        "train_metrics": train_metrics,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "details": details,
        # Metric rationale:
        # We optimize macro-F1 because classes are imbalanced and we want each
        # activity to matter equally, not just majority classes.
        "metric_rationale": (
            "Macro-F1 used as primary CV metric to balance precision/recall "
            "per class under class imbalance; weighted-F1 and accuracy are "
            "reported as secondary diagnostics."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return model_path, meta_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train final XGBoost HAR model.")
    parser.add_argument(
        "--api",
        choices=["sklearn", "native"],
        default="sklearn",
        help="Training API branch to use.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help=f"Lazy sampling fraction before collect (default: {DEFAULT_SAMPLE_FRACTION}).",
    )
    parser.add_argument(
        "--target-column",
        choices=["activity", "activity_general"],
        default="activity",
        help="Target label column to train on.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"K-Fold CV folds (default: {DEFAULT_N_FOLDS}).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Random search trials per branch (default: {DEFAULT_N_TRIALS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--reserve-threads",
        type=int,
        default=DEFAULT_RESERVE_THREADS,
        help=(
            "Leave this many CPU threads free for your machine; "
            "trainer uses max(1, cpu_count - reserve_threads)."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="XGBoost verbosity (0=silent, 1=warning, 2=info, 3=debug).",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end training."""
    args = parse_args()
    if not 0 < args.sample_fraction <= 1:
        msg = "--sample-fraction must be in (0, 1]."
        raise ValueError(msg)
    if args.n_folds < 2:
        msg = "--n-folds must be >= 2."
        raise ValueError(msg)

    train_threads = compute_training_threads(args.reserve_threads)
    print("Loading data with lazy sample...")
    print(f"sample_fraction={args.sample_fraction:.3f}, api={args.api}, target={args.target_column}")
    print(f"training_threads={train_threads} (reserving {args.reserve_threads})")

    df = load_sampled_dataset(sample_fraction=args.sample_fraction, seed=args.seed)
    X, y_raw, y_encoded, feature_columns, encoder = select_features(
        df=df, target_column=args.target_column
    )
    print(f"sampled_rows={df.height:,}, features={len(feature_columns)}, classes={len(encoder.classes_)}")

    # Primary metric choice:
    # Macro-F1 is the optimization target because the class distribution is
    # imbalanced and we want balanced performance across activities.
    metric_name = "f1_macro"

    if args.api == "native":
        model, best_cv_metric, details = train_native_api(
            X=X,
            y=y_encoded,
            n_folds=args.n_folds,
            n_trials=args.n_trials,
            seed=args.seed,
            train_threads=train_threads,
            verbosity=args.verbosity,
        )
    else:
        model, best_cv_metric, details = train_sklearn_api(
            X=X,
            y=y_raw,
            n_folds=args.n_folds,
            n_trials=args.n_trials,
            seed=args.seed,
            train_threads=train_threads,
            verbosity=args.verbosity,
            class_labels=encoder.classes_,
        )

    y_for_metrics: np.ndarray = y_encoded if args.api == "native" else y_raw
    train_metrics = compute_train_metrics(model=model, X=X, y=y_for_metrics)
    model_path, meta_path = save_outputs(
        model=model,
        details=details,
        metric_value=best_cv_metric,
        metric_name=metric_name,
        api=args.api,
        target_column=args.target_column,
        feature_columns=feature_columns,
        label_encoder=encoder,
        sample_fraction=args.sample_fraction,
        train_threads=train_threads,
        train_metrics=train_metrics,
    )

    print("\nTraining complete.")
    print(f"primary_cv_{metric_name}={best_cv_metric:.5f}")
    print(f"train_f1_macro={train_metrics['train_f1_macro']:.5f}")
    print(f"saved_model={model_path}")
    print(f"saved_metadata={meta_path}")


if __name__ == "__main__":
    main()
