<!-- AUTO-GENERATED: docs/executive_summary/tools/generate_model_metrics_md.py -->
<!-- Generated: 2026-04-09 23:00:13 MDT -->

### Best model (current leader)

- Model: **XGBoost native**
- Target: **Generalized activities (11 classes)**
- CV macro-F1: **0.7772**
- Sample fraction: **0.4**
- Train macro-F1: 0.8088; train accuracy: 0.8304
- Boosting: 900 rounds (early_stopping_rounds=40)
- Key settings: learning_rate=0.030, max_depth=6, rounds=900

_Showing top 5 runs (by CV macro-F1)._

| Model | Target | Sample fraction | Macro-F1 (CV) | Macro-F1 (train) | Accuracy (train) |
|---|---|---:|---:|---:|---:|
| **XGBoost native** | **Generalized activities (11 classes)** | 0.4 | **0.7772** | 0.8088 | 0.8304 |
| XGBoost sklearn | Generalized activities (11 classes) | 0.01 | 0.6221 | 0.7607 | 0.8638 |
| XGBoost sklearn | Generalized activities (11 classes) | 0.01 | 0.6157 | 0.7607 | 0.8638 |
| XGBoost native | Generalized activities (11 classes) | 0.005 | 0.5938 | 0.7583 | 0.8563 |
| XGBoost native | Generalized activities (11 classes) | 0.005 | 0.5829 | 0.7615 | 0.8591 |
