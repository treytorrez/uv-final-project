#!/usr/bin/env python3
"""
Beta Baseline Training Script for CASAS HAR Project

This script runs the complete training pipeline overnight:
1. Loads all CASAS homes (with configurable sampling)
2. Applies preprocessing (cyclic time, feature interactions)
3. Runs leave-one-out CV with early stopping
4. Saves results, models, and analysis

Usage:
    # Full overnight run
    python scripts/train_beta_baseline.py
    
    # Quick test (3 homes, 5% sample)
    python scripts/train_beta_baseline.py --quick-test
    
    # Custom sample fraction
    python scripts/train_beta_baseline.py --sample 0.1

Expected runtime:
    - Quick test: ~15-30 minutes
    - Full run (20% sample): 4-6 hours
"""

import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl

from ds_project.data.loaders import load_all_casas_homes, get_available_homes
from ds_project.features.transforms import preprocess_casas_features
from ds_project.models.training import leave_one_home_out_cv, CVResults, FoldResult

from beta_config import BetaConfig, beta_config


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(config: BetaConfig) -> logging.Logger:
    """Configure logging for the training run."""
    log_file = config.results_path / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_cv_results(results: CVResults, config: BetaConfig, logger: logging.Logger) -> None:
    """Save all cross-validation results to disk."""
    output_dir = config.results_path
    
    # 1. Summary JSON (human-readable aggregate results)
    summary = {
        "experiment_name": results.experiment_name,
        "timestamp": results.timestamp,
        "config": {
            "sample_fraction": config.sample_fraction,
            "xgboost_params": config.xgboost_params,
            "early_stopping_rounds": config.early_stopping_rounds,
            "include_cyclic_time": config.include_cyclic_time,
            "include_feature_interactions": config.include_feature_interactions,
        },
        "overall": {
            "mean_accuracy": float(results.mean_accuracy),
            "std_accuracy": float(results.std_accuracy),
            "min_accuracy": float(results.min_accuracy),
            "max_accuracy": float(results.max_accuracy),
            "mean_f1_macro": float(results.mean_f1_macro),
            "total_training_time_minutes": float(results.total_training_time / 60),
            "n_homes": len(results.fold_results),
            "n_features": len(results.feature_names),
        },
        "per_home": {
            home_id: {
                "accuracy": float(fold.accuracy),
                "f1_macro": float(fold.f1_macro),
                "f1_weighted": float(fold.f1_weighted),
                "n_train_samples": fold.n_train_samples,
                "n_test_samples": fold.n_test_samples,
                "n_trees_used": fold.n_trees_used,
                "training_time_seconds": float(fold.training_time_seconds),
            }
            for home_id, fold in results.fold_results.items()
        },
        "classes": list(results.label_encoder.classes_),
        "features": results.feature_names,
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    # 2. Detailed results pickle (for further analysis)
    detailed_path = output_dir / "detailed_results.pkl"
    with open(detailed_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved detailed results: {detailed_path}")
    
    # 3. Feature importance aggregated
    if config.save_feature_importance:
        save_feature_importance(results, output_dir, logger)
    
    # 4. Confusion matrices
    if config.save_confusion_matrices:
        save_confusion_matrices(results, output_dir, logger)
    
    # 5. Per-home predictions
    if config.save_predictions:
        save_predictions(results, output_dir, logger)
    
    # 6. Plain text summary
    summary_txt_path = output_dir / "summary.txt"
    with open(summary_txt_path, 'w') as f:
        f.write(results.summary())
    logger.info(f"Saved text summary: {summary_txt_path}")


def save_feature_importance(results: CVResults, output_dir: Path, logger: logging.Logger) -> None:
    """Save aggregated feature importance across all folds."""
    # Aggregate feature importance
    importance_agg: Dict[str, list] = {f: [] for f in results.feature_names}
    
    for fold in results.fold_results.values():
        if fold.feature_importance:
            for feature, importance in fold.feature_importance.items():
                if feature in importance_agg:
                    importance_agg[feature].append(importance)
    
    # Calculate mean and std
    importance_stats = {
        feature: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for feature, values in importance_agg.items()
        if values
    }
    
    # Sort by mean importance
    sorted_importance = dict(
        sorted(importance_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    )
    
    importance_path = output_dir / "feature_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    logger.info(f"Saved feature importance: {importance_path}")
    
    # Also save top features as plain text
    top_features_path = output_dir / "top_features.txt"
    with open(top_features_path, 'w') as f:
        f.write("TOP 20 FEATURES BY IMPORTANCE\n")
        f.write("="*50 + "\n\n")
        for i, (feature, stats) in enumerate(list(sorted_importance.items())[:20], 1):
            f.write(f"{i:2d}. {feature:40s} mean={stats['mean']:.4f} ± {stats['std']:.4f}\n")
    logger.info(f"Saved top features: {top_features_path}")


def save_confusion_matrices(results: CVResults, output_dir: Path, logger: logging.Logger) -> None:
    """Save confusion matrices for all folds."""
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    classes = list(results.label_encoder.classes_)
    
    # Save individual fold confusion matrices
    for home_id, fold in results.fold_results.items():
        if fold.confusion_matrix is not None:
            cm_path = cm_dir / f"{home_id}_confusion_matrix.csv"
            
            # Create DataFrame with class labels
            cm_df = pl.DataFrame(
                fold.confusion_matrix,
                schema={f"pred_{c}": pl.Int64 for c in classes}
            ).with_columns(
                pl.Series("actual", classes)
            ).select(["actual"] + [f"pred_{c}" for c in classes])
            
            cm_df.write_csv(cm_path)
    
    # Save aggregated confusion matrix
    agg_cm = np.zeros_like(list(results.fold_results.values())[0].confusion_matrix)
    for fold in results.fold_results.values():
        if fold.confusion_matrix is not None:
            agg_cm += fold.confusion_matrix
    
    agg_cm_df = pl.DataFrame(
        agg_cm,
        schema={f"pred_{c}": pl.Int64 for c in classes}
    ).with_columns(
        pl.Series("actual", classes)
    ).select(["actual"] + [f"pred_{c}" for c in classes])
    
    agg_cm_path = output_dir / "aggregated_confusion_matrix.csv"
    agg_cm_df.write_csv(agg_cm_path)
    logger.info(f"Saved confusion matrices: {cm_dir}")


def save_predictions(results: CVResults, output_dir: Path, logger: logging.Logger) -> None:
    """Save predictions for error analysis."""
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    classes = list(results.label_encoder.classes_)
    
    for home_id, fold in results.fold_results.items():
        pred_df = pl.DataFrame({
            "y_true": results.label_encoder.inverse_transform(fold.y_true),
            "y_pred": results.label_encoder.inverse_transform(fold.y_pred),
            "correct": fold.y_true == fold.y_pred,
        })
        
        # Add probability columns
        if fold.y_pred_proba is not None:
            for i, cls in enumerate(classes):
                pred_df = pred_df.with_columns(
                    pl.Series(f"prob_{cls}", fold.y_pred_proba[:, i])
                )
        
        pred_path = pred_dir / f"{home_id}_predictions.parquet"
        pred_df.write_parquet(pred_path)
    
    logger.info(f"Saved predictions: {pred_dir}")


def save_checkpoint(home_id: str, fold_result: FoldResult, config: BetaConfig) -> None:
    """Save checkpoint after each fold."""
    checkpoint_path = config.checkpoint_dir / f"checkpoint_{home_id}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(fold_result, f)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_beta_training(
    config: BetaConfig,
    quick_test: bool = False,
    sample_override: Optional[float] = None,
    homes_override: Optional[int] = None
) -> CVResults:
    """Run the complete beta training pipeline.
    
    Args:
        config: Beta configuration
        quick_test: If True, run with reduced data for testing
        sample_override: Override sample fraction
        homes_override: Limit number of homes to process
        
    Returns:
        CVResults with all training results
    """
    # Setup
    logger = setup_logging(config)
    
    # Apply overrides
    sample_fraction = sample_override or config.sample_fraction
    if quick_test:
        sample_fraction = 0.05
        homes_override = homes_override or 3
    
    logger.info("="*70)
    logger.info("BETA BASELINE TRAINING")
    logger.info("="*70)
    logger.info(f"Experiment: {config.experiment_id}")
    logger.info(f"Sample fraction: {sample_fraction}")
    logger.info(f"Early stopping: {config.early_stopping_rounds} rounds")
    logger.info(f"Learning rate: {config.xgboost_params['learning_rate']}")
    logger.info(f"Max trees: {config.xgboost_params['n_estimators']}")
    logger.info(f"Output directory: {config.results_path}")
    logger.info("="*70)
    
    # ───────────────────────────────────────────────────────────────────────────
    # PHASE 1: Load Data
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n[PHASE 1] Loading CASAS data...")
    phase_start = time.time()
    
    available_homes = get_available_homes(config.data_dir)
    logger.info(f"Found {len(available_homes)} homes")
    
    # Optionally limit homes
    if homes_override:
        available_homes = available_homes[:homes_override]
        logger.info(f"Limited to {len(available_homes)} homes for testing")
    
    # Load data
    lf = load_all_casas_homes(
        data_dir=config.data_dir,
        home_range=config.home_range
    )
    
    # Apply sampling by collecting with limit
    # Note: For proper sampling, we'd need to collect first, but for memory
    # efficiency with large data, we'll sample after preprocessing
    
    logger.info(f"Phase 1 complete: {time.time() - phase_start:.1f}s")
    
    # ───────────────────────────────────────────────────────────────────────────
    # PHASE 2: Preprocessing
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n[PHASE 2] Preprocessing features...")
    phase_start = time.time()
    
    lf_processed = preprocess_casas_features(
        lf,
        include_cyclic=config.include_cyclic_time,
        include_interactions=config.include_feature_interactions
    )
    
    # Filter to available homes only (if limited)
    if homes_override:
        lf_processed = lf_processed.filter(pl.col("home_id").is_in(available_homes))
    
    # Sample the data
    if sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction*100:.0f}% of data...")
        # Collect to DataFrame for sampling
        df_full = lf_processed.collect()
        n_total = len(df_full)
        df_sampled = df_full.sample(fraction=sample_fraction, seed=config.random_seed)
        n_sampled = len(df_sampled)
        logger.info(f"Sampled {n_sampled:,} from {n_total:,} total samples")
        lf_processed = df_sampled.lazy()
    
    # Get feature info
    schema = lf_processed.schema
    n_features = sum(1 for col, dtype in schema.items() 
                     if dtype.is_numeric() and col not in ['home_id', 'activity_grouped', 'activity_original', 'activity'])
    logger.info(f"Features after preprocessing: {n_features}")
    logger.info(f"Phase 2 complete: {time.time() - phase_start:.1f}s")
    
    # ───────────────────────────────────────────────────────────────────────────
    # PHASE 3: Cross-Validation Training
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n[PHASE 3] Starting leave-one-out cross-validation...")
    phase_start = time.time()
    
    # Progress callback
    def progress_callback(fold_num: int, total_folds: int, home_id: str, accuracy: float):
        elapsed = time.time() - phase_start
        avg_time_per_fold = elapsed / fold_num
        remaining = avg_time_per_fold * (total_folds - fold_num)
        logger.info(
            f"Progress: {fold_num}/{total_folds} ({fold_num/total_folds*100:.0f}%) | "
            f"ETA: {remaining/60:.1f} min"
        )
    
    # Checkpoint callback
    def checkpoint_callback(home_id: str, fold_result: FoldResult):
        if config.checkpoint_interval > 0:
            fold_num = len([f for f in config.checkpoint_dir.glob("checkpoint_*.pkl")])
            if fold_num % config.checkpoint_interval == 0:
                save_checkpoint(home_id, fold_result, config)
    
    # Run CV
    cv_results = leave_one_home_out_cv(
        lf_processed,
        target_col="activity_grouped",
        params=config.xgboost_params,
        early_stopping_rounds=config.early_stopping_rounds,
        eval_fraction=config.eval_fraction,
        experiment_name=config.experiment_id,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
        verbose=True
    )
    
    logger.info(f"Phase 3 complete: {time.time() - phase_start:.1f}s")
    
    # ───────────────────────────────────────────────────────────────────────────
    # PHASE 4: Save Results
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n[PHASE 4] Saving results...")
    phase_start = time.time()
    
    save_cv_results(cv_results, config, logger)
    
    logger.info(f"Phase 4 complete: {time.time() - phase_start:.1f}s")
    
    # ───────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ───────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Results saved to: {config.results_path}")
    logger.info(f"Mean accuracy: {cv_results.mean_accuracy:.4f} ± {cv_results.std_accuracy:.4f}")
    logger.info(f"Mean F1 (macro): {cv_results.mean_f1_macro:.4f}")
    logger.info(f"Total time: {cv_results.total_training_time/60:.1f} minutes")
    logger.info("="*70)
    
    return cv_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train beta baseline model for CASAS HAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full overnight run (20% sample, all 30 homes)
    python scripts/train_beta_baseline.py
    
    # Quick test (5% sample, 3 homes)
    python scripts/train_beta_baseline.py --quick-test
    
    # Custom sample fraction
    python scripts/train_beta_baseline.py --sample 0.1
    
    # Limit homes for faster iteration
    python scripts/train_beta_baseline.py --homes 10 --sample 0.1
"""
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 5%% sample and 3 homes"
    )
    
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Override sample fraction (0.0-1.0)"
    )
    
    parser.add_argument(
        "--homes",
        type=int,
        default=None,
        help="Limit number of homes to process"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom experiment name"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = BetaConfig()
    
    if args.name:
        config.experiment_name = args.name
    
    # Run training
    try:
        cv_results = run_beta_training(
            config,
            quick_test=args.quick_test,
            sample_override=args.sample,
            homes_override=args.homes
        )
        
        print("\n✅ Training completed successfully!")
        print(f"   Results: {config.results_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Partial results may be available in checkpoints")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
