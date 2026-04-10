<!-- AUTO-GENERATED: docs/executive_summary/tools/generate_model_metrics_md.py -->
<!-- Generated: 2026-04-09 22:44:25 MDT -->

### Best model (current leader)

- Model: **XGBoost native**
- Target: **Generalized activities (11 classes)**
- CV macro-F1: **0.7772** (folds=2)
- Data fraction: 0.4
- Train macro-F1: 0.8088; train accuracy: 0.8304
- Boosting: 900 rounds (early_stopping_rounds=40)
- Key settings: learning_rate=0.030, max_depth=6, rounds=900

| Model | Target | Macro-F1 (CV) | Macro-F1 (train) | Accuracy (train) |
|---|---|---:|---:|---:|
| XGBoost native | Generalized activities (11 classes) | **0.7772** | 0.8088 | 0.8304 |
| XGBoost sklearn | Generalized activities (11 classes) | 0.6221 | 0.7607 | 0.8638 |
| XGBoost sklearn | Generalized activities (11 classes) | 0.6157 | 0.7607 | 0.8638 |
| XGBoost native | Generalized activities (11 classes) | 0.5938 | 0.7583 | 0.8563 |
| XGBoost native | Generalized activities (11 classes) | 0.5829 | 0.7615 | 0.8591 |
| XGBoost sklearn | Generalized activities (10 classes) | 0.4814 | 0.9295 | 0.9882 |
| XGBoost native | Generalized activities (10 classes) | 0.4445 | 0.5499 | 0.7474 |
| XGBoost sklearn | Generalized activities (11 classes) | N/A | 0.7607 | 0.8638 |
