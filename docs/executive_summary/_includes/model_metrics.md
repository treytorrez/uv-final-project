<!-- AUTO-GENERATED: docs/executive_summary/tools/generate_model_metrics_md.py -->
<!-- Generated: 2026-04-09 22:00:09 MDT -->

### Best model (current leader)

- Run: `20260409_214833` (`native` / `activity_general`, 11 classes, sample_fraction=0.4)
- CV macro-F1: `0.7772` (folds=2)
- Train macro-F1: `0.8088`; train accuracy: `0.8304`
- Boosting: `900` rounds (early_stopping_rounds=40)
- Key params: learning_rate=0.030, max_depth=6, min_child_weight=3, subsample=0.800, colsample_bytree=0.800, alpha=0.100, lambda=0.800

> Notes to include (edit in the QMD): why this run is the pick, what split was used, and what the next validation step is.

| timestamp | api | target | sample_fraction | threads | cv_folds | cv_macro_f1 | train_macro_f1 | train_accuracy | n_classes | meta_file |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20260409_215520` | `sklearn` | `activity_general` | 0.001 | 12 | N/A | 0.4814 | 0.9295 | 0.9882 | 10 | `meta_sklearn_activity_general_f1_macro_0.4814_20260409_215520.json` |
| `20260409_215447` | `native` | `activity_general` | 0.001 | 12 | 2 | 0.4445 | 0.5499 | 0.7474 | 10 | `meta_native_activity_general_f1_macro_0.4445_20260409_215447.json` |
| `20260409_214833` | `native` | `activity_general` | 0.4 | 15 | 2 | 0.7772 | 0.8088 | 0.8304 | 11 | `meta_native_activity_general_f1_macro_0.7772_20260409_214833.json` |
| `20260408_221512` | `native` | `activity_general` | 0.005 | 12 | N/A | 0.5829 | 0.7615 | 0.8591 | 11 | `meta_native_activity_general_f1_macro_0.5829_20260408_221512.json` |
| `20260408_221358` | `sklearn` | `activity_general` | 0.01 | 12 | N/A | 0.6221 | 0.7607 | 0.8638 | 11 | `meta_sklearn_activity_general_f1_macro_0.6221_20260408_221358.json` |
| `20260408_221101` | `sklearn` | `activity_general` | 0.01 | 12 | N/A | N/A | 0.7607 | 0.8638 | 11 | `meta_sklearn_activity_general_f1_macro_nan_20260408_221101.json` |
| `20260408_215912` | `native` | `activity_general` | 0.005 | 12 | N/A | 0.5938 | 0.7583 | 0.8563 | 11 | `meta_native_activity_general_f1_macro_0.5938_20260408_215912.json` |
| `20260408_215753` | `sklearn` | `activity_general` | 0.01 | 12 | N/A | 0.6157 | 0.7607 | 0.8638 | 11 | `meta_sklearn_activity_general_f1_macro_0.6157_20260408_215753.json` |
