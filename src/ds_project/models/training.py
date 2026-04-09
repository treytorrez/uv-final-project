"""Model training functions for CASAS HAR classification.

Implements XGBoost-based activity recognition with leave-one-out cross-validation
across smart homes. Supports early stopping, progress logging, and checkpointing.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

import numpy as np
import polars as pl
from polars import LazyFrame, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import xgboost as xgb

from ..config import config

# Configure logging
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Results from a single CV fold (one home held out)."""
    home_id: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    n_train_samples: int
    n_test_samples: int
    n_trees_used: int
    training_time_seconds: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_proba: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


@dataclass
class CVResults:
    """Complete cross-validation results."""
    experiment_name: str
    timestamp: str
    fold_results: Dict[str, FoldResult]
    label_encoder: LabelEncoder
    feature_names: List[str]
    config_used: Dict[str, Any]
    
    # Aggregate statistics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    mean_f1_macro: float = 0.0
    total_training_time: float = 0.0
    
    def __post_init__(self):
        """Calculate aggregate statistics."""
        if self.fold_results:
            accuracies = [r.accuracy for r in self.fold_results.values()]
            f1_scores = [r.f1_macro for r in self.fold_results.values()]
            times = [r.training_time_seconds for r in self.fold_results.values()]
            
            self.mean_accuracy = np.mean(accuracies)
            self.std_accuracy = np.std(accuracies)
            self.min_accuracy = np.min(accuracies)
            self.max_accuracy = np.max(accuracies)
            self.mean_f1_macro = np.mean(f1_scores)
            self.total_training_time = np.sum(times)
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        return f"""
═══════════════════════════════════════════════════════════════════════════════
CROSS-VALIDATION RESULTS: {self.experiment_name}
═══════════════════════════════════════════════════════════════════════════════

Timestamp: {self.timestamp}
Homes evaluated: {len(self.fold_results)}
Total training time: {self.total_training_time/60:.1f} minutes

ACCURACY:
  Mean:  {self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}
  Range: {self.min_accuracy:.4f} - {self.max_accuracy:.4f}

F1 SCORE (macro):
  Mean:  {self.mean_f1_macro:.4f}

FEATURES: {len(self.feature_names)}
CLASSES:  {list(self.label_encoder.classes_)}

PER-HOME RESULTS:
{self._format_per_home_results()}
═══════════════════════════════════════════════════════════════════════════════
"""
    
    def _format_per_home_results(self) -> str:
        """Format per-home results table."""
        lines = []
        sorted_homes = sorted(
            self.fold_results.items(), 
            key=lambda x: x[1].accuracy, 
            reverse=True
        )
        for home_id, result in sorted_homes:
            lines.append(
                f"  {home_id}: acc={result.accuracy:.3f}, "
                f"f1={result.f1_macro:.3f}, "
                f"n_test={result.n_test_samples:,}, "
                f"trees={result.n_trees_used}, "
                f"time={result.training_time_seconds:.1f}s"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_training_data(
    lf: LazyFrame, 
    target_col: str = "activity_grouped",
    exclude_features: List[str] | None = None,
    label_encoder: LabelEncoder | None = None
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """Prepare data for XGBoost training.
    
    Collects the LazyFrame and converts to numpy arrays suitable for sklearn/XGBoost.
    Encodes string labels to integers and returns feature names and label encoder.
    
    Args:
        lf: LazyFrame with preprocessed features
        target_col: Name of target column to predict
        exclude_features: Additional features to exclude from training
        label_encoder: Pre-fitted LabelEncoder (for test data consistency)
        
    Returns:
        Tuple of (X, y, feature_names, label_encoder)
    """
    # Collect the LazyFrame to DataFrame
    df = lf.collect()
    
    # Identify features to exclude
    exclude_cols = {"home_id", target_col, "activity_original", "activity"}
    if exclude_features:
        exclude_cols.update(exclude_features)
    
    # Get feature columns (only numeric)
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols and df[col].dtype.is_numeric()
    ]
    
    # Extract features and target
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select(target_col).to_numpy().ravel()
    
    # Encode string labels to integers
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)
    
    return X, y, feature_cols, label_encoder


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    early_stopping_rounds: int = 100,
    verbose: bool = True
) -> Tuple[xgb.XGBClassifier, int]:
    """Train XGBoost with early stopping.
    
    Uses a validation set to determine when to stop training, preventing
    overfitting while allowing the model to train longer with lower learning rates.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features  
        y_val: Validation labels
        params: XGBoost parameters
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, number of trees used)
    """
    # Remove early stopping from params if present (we'll set it separately)
    params = params.copy()
    params.pop('early_stopping_rounds', None)
    
    model = xgb.XGBClassifier(
        early_stopping_rounds=early_stopping_rounds,
        **params
    )
    
    # Train with validation set for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose if verbose else False
    )
    
    n_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params.get('n_estimators', 100)
    
    return model, n_trees


def train_xgboost_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any] | None = None,
    eval_fraction: float = 0.1,
    early_stopping_rounds: int = 100,
    verbose: bool = False
) -> Tuple[xgb.XGBClassifier, int]:
    """Train XGBoost classifier with optional early stopping.
    
    Args:
        X: Feature matrix
        y: Target vector (integer encoded)
        params: XGBoost parameters (uses config defaults if None)
        eval_fraction: Fraction of training data for early stopping validation
        early_stopping_rounds: Stop if no improvement for N rounds
        verbose: Whether to print training progress
        
    Returns:
        Tuple of (trained model, number of trees used)
    """
    if params is None:
        params = config.xgboost_params.copy()
    
    # Split for early stopping
    if eval_fraction > 0 and early_stopping_rounds > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=eval_fraction, random_state=42, stratify=y
        )
        return train_xgboost_with_early_stopping(
            X_train, y_train, X_val, y_val,
            params, early_stopping_rounds, verbose
        )
    else:
        # No early stopping
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        return model, params.get('n_estimators', 100)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def leave_one_home_out_cv(
    lf: LazyFrame,
    target_col: str = "activity_grouped",
    params: Dict[str, Any] | None = None,
    early_stopping_rounds: int = 100,
    eval_fraction: float = 0.1,
    experiment_name: str = "cv_experiment",
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
    checkpoint_callback: Optional[Callable[[str, FoldResult], None]] = None,
    verbose: bool = True
) -> CVResults:
    """Perform leave-one-home-out cross-validation with early stopping.
    
    For each home, trains on all other homes and tests on that home.
    This evaluates model generalization across different smart home environments.
    
    Args:
        lf: LazyFrame with preprocessed features and home_id column
        target_col: Target column name
        params: XGBoost parameters (uses config defaults if None)
        early_stopping_rounds: Stop if no improvement for N rounds
        eval_fraction: Fraction of training data for early stopping
        experiment_name: Name for this experiment run
        progress_callback: Optional callback(fold_num, total_folds, home_id, accuracy)
        checkpoint_callback: Optional callback(home_id, fold_result) for saving
        verbose: Whether to print progress
        
    Returns:
        CVResults object with all fold results and statistics
    """
    if params is None:
        params = config.xgboost_params.copy()
    
    # Get all homes
    all_homes = sorted(lf.select("home_id").unique().collect()["home_id"].to_list())
    n_homes = len(all_homes)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if verbose:
        logger.info(f"Starting leave-one-out CV across {n_homes} homes")
        print(f"\n{'═'*70}")
        print(f"LEAVE-ONE-OUT CROSS-VALIDATION")
        print(f"Homes: {n_homes} | Early stopping: {early_stopping_rounds} rounds")
        print(f"{'═'*70}\n")
    
    fold_results: Dict[str, FoldResult] = {}
    label_encoder: Optional[LabelEncoder] = None
    feature_names: List[str] = []
    
    total_start_time = time.time()
    
    for i, test_home in enumerate(all_homes, 1):
        fold_start_time = time.time()
        
        if verbose:
            print(f"[{i:2d}/{n_homes}] Training with {test_home} held out...", end=" ", flush=True)
        
        # Split data
        train_lf = lf.filter(pl.col("home_id") != test_home)
        test_lf = lf.filter(pl.col("home_id") == test_home)
        
        # Prepare training data
        X_train, y_train, feature_names, label_encoder = prepare_training_data(
            train_lf, target_col, label_encoder=label_encoder
        )
        
        # Prepare test data (using same label encoder)
        X_test, y_test, _, _ = prepare_training_data(
            test_lf, target_col, label_encoder=label_encoder
        )
        
        # Train model with early stopping
        model, n_trees = train_xgboost_model(
            X_train, y_train,
            params=params,
            eval_fraction=eval_fraction,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Feature importance
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
        
        fold_time = time.time() - fold_start_time
        
        # Create fold result
        fold_result = FoldResult(
            home_id=test_home,
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
            n_trees_used=n_trees,
            training_time_seconds=fold_time,
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            feature_importance=importance_dict,
            confusion_matrix=cm,
            classification_report=class_report
        )
        
        fold_results[test_home] = fold_result
        
        if verbose:
            print(f"acc={accuracy:.3f}, f1={f1_macro:.3f}, trees={n_trees}, time={fold_time:.1f}s")
        
        # Callbacks
        if progress_callback:
            progress_callback(i, n_homes, test_home, accuracy)
        
        if checkpoint_callback:
            checkpoint_callback(test_home, fold_result)
    
    total_time = time.time() - total_start_time
    
    # Create final results
    cv_results = CVResults(
        experiment_name=experiment_name,
        timestamp=timestamp,
        fold_results=fold_results,
        label_encoder=label_encoder,
        feature_names=feature_names,
        config_used=params
    )
    
    if verbose:
        print(cv_results.summary())
    
    return cv_results


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def leave_one_home_out_cv_simple(
    lf: LazyFrame,
    target_col: str = "activity_grouped"
) -> Dict[str, Dict[str, float]]:
    """Simple leave-one-out CV (legacy interface).
    
    Provides backward compatibility with the original simple interface.
    """
    cv_results = leave_one_home_out_cv(
        lf, target_col,
        early_stopping_rounds=0,
        eval_fraction=0,
        verbose=True
    )
    
    # Convert to simple dict format
    results = {}
    for home_id, fold in cv_results.fold_results.items():
        results[home_id] = {
            "accuracy": fold.accuracy,
            "n_train_samples": fold.n_train_samples,
            "n_test_samples": fold.n_test_samples,
            "n_classes": len(cv_results.label_encoder.classes_)
        }
    
    results["overall"] = {
        "mean_accuracy": cv_results.mean_accuracy,
        "std_accuracy": cv_results.std_accuracy,
        "min_accuracy": cv_results.min_accuracy,
        "max_accuracy": cv_results.max_accuracy,
        "n_homes": len(cv_results.fold_results)
    }
    
    return results
