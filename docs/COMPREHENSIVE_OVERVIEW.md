# CASAS HAR Project - Comprehensive Code Overview

## Executive Summary

This project implements a complete machine learning pipeline for Human Activity Recognition (HAR) using the CASAS smart home dataset. The system classifies daily activities from ambient sensor data across 30 smart homes, with emphasis on generalization via leave-one-out cross-validation.

**Key Features:**
- **LazyFrame-first processing** for memory efficiency with large datasets (~14M samples)
- **Activity grouping** reduces 35 raw activities to 11 meaningful categories
- **Leave-one-out CV** tests model generalization across different homes
- **XGBoost classification** handles sparse sensor features and class imbalance
- **Type-safe configuration** using pydantic-settings
- **Modular architecture** with clear separation of concerns

## Architecture Overview

The codebase is organized into 5 main modules:

```
src/ds_project/
├── config.py           # Centralized configuration (17 parameters)
├── data/               # Loading & schemas (3 functions + 2 schemas)
│   ├── loaders.py      # CASAS data loading functions
│   └── schemas.py      # 37-feature schema definitions
├── features/           # Transformations (4 functions + activity mapping)
│   └── transforms.py   # Preprocessing pipeline
├── models/             # Training & evaluation (3 functions)
│   └── training.py     # XGBoost + leave-one-out CV
└── io/                 # Persistence (7 functions)
    └── persistence.py  # Model/results saving
```

**Data Flow:**
```
Raw CSV Files → Load (data/) → Transform (features/) → Train (models/) → Save (io/)
```

---

## Core Modules

### 1. Configuration (`config.py`)

Central configuration management with 17 parameters using pydantic-settings:

```python
from ds_project.config import config

# Access configured values
print(f"Sample fraction: {config.casas_sample_fraction}")  # 0.2
print(f"XGBoost params: {config.xgboost_params}")
print(f"Activity groups: {len(config.grouped_activities)}")  # 11

# Environment variable support (HAR_ prefix)
# export HAR_CASAS_SAMPLE_FRACTION=0.5
# config.casas_sample_fraction will now be 0.5
```

**Key Configuration Categories:**
- **Paths**: All directory locations (data/, models/, reports/, etc.)
- **Dataset**: CASAS home range (101-130), sample fraction (20%)
- **Activity Grouping**: 35→11 class mappings
- **Model**: XGBoost hyperparameters, CV strategy
- **Compute**: Memory limits, CPU cores

### 2. Data Loading (`data/`)

#### `data/schemas.py`
Defines the complete 37-feature schema for CASAS dataset:

```python
from ds_project.data.schemas import CASAS_FEATURES_SCHEMA

# Schema breakdown:
# - Temporal features (3): lastSensorEventHours, lastSensorEventSeconds, lastSensorDayOfWeek
# - Window metadata (5): windowDuration, complexity, etc.
# - Sensor IDs (5): prevDominantSensor1/2, lastSensorID, etc.
# - Sensor counts (11): sensorCount-{Bathroom,Kitchen,LivingRoom,...}
# - Elapsed times (11): sensorElTime-{Bathroom,Kitchen,LivingRoom,...}
# - Constant feature (1): numDistinctSensors (always 0, will be dropped)
# - Target (1): activity (35 unique values)

print(f"Total features: {len(CASAS_FEATURES_SCHEMA)}")  # 37
```

#### `data/loaders.py`
Three functions for loading CASAS data:

```python
from ds_project.data.loaders import load_casas_home, load_all_casas_homes, get_available_homes

# 1. Check what data is available
homes = get_available_homes()  # ['csh101', 'csh102', ..., 'csh130']
print(f"Found {len(homes)} homes")

# 2. Load single home (returns LazyFrame)
lf = load_casas_home('csh101')
print(f"Home csh101 schema: {lf.schema}")

# 3. Load all homes combined
all_homes_lf = load_all_casas_homes(
    home_range=(101, 130),  # csh101 to csh130
    sample_fraction=0.2     # Use 20% of each home's data
)
```

**Why LazyFrame?** Polars LazyFrame allows us to build complex transformation pipelines without loading all 14M samples into memory. Operations are deferred until `.collect()` is called.

### 3. Feature Engineering (`features/`)

#### `features/transforms.py`
Four transformation functions implementing the preprocessing pipeline:

```python
from ds_project.features.transforms import (
    apply_activity_grouping, 
    drop_constant_features,
    add_cyclic_time_features,
    preprocess_casas_features  # Complete pipeline
)

# Load and preprocess data
lf = load_casas_home('csh101')
lf_processed = preprocess_casas_features(lf, include_cyclic=True)

# Check what changed
print(f"Original features: {len(lf.schema)}")        # 37
print(f"Processed features: {len(lf_processed.schema)}")  # 41 (added 4 cyclic features)

# View sample of processed data
sample = lf_processed.head(5).collect()
print(sample.select(['activity_original', 'activity_grouped', 'hour_sin', 'hour_cos']))
```

**Activity Grouping (35 → 11 classes):**
The transformation reduces class complexity by grouping related activities:

```python
# Original: Cook_Breakfast, Cook_Lunch, Cook_Dinner, Cook → Grouped: Cooking
# Original: Sleep, Sleep_Out_Of_Bed, Bed_Toilet_Transition → Grouped: Sleeping
# Original: Bathe, Toilet, Personal_Hygiene, Groom, Dress → Grouped: Personal_Care

from ds_project.features.transforms import GROUPED_ACTIVITIES
for group, activities in GROUPED_ACTIVITIES.items():
    print(f"{group}: {activities}")
```

**Why Activity Grouping?** The original 35 classes create severe class imbalance (some activities <1% of data). Grouping creates more balanced classes while preserving semantic meaning.

**Cyclic Time Features:** Converts linear time (hours: 0-23) to circular representation using sine/cosine. This helps the model understand that hour 23 is close to hour 0.

### 4. Model Training (`models/`)

#### `models/training.py`
Three functions implementing XGBoost training with leave-one-out CV:

```python
from ds_project.models.training import (
    prepare_training_data,
    train_xgboost_model, 
    leave_one_home_out_cv
)

# Complete training workflow
lf_processed = preprocess_casas_features(load_all_casas_homes())

# 1. Prepare data for sklearn/XGBoost
X, y, feature_names, label_encoder = prepare_training_data(
    lf_processed, 
    target_col="activity_grouped"
)
print(f"Training matrix: {X.shape}")  # (n_samples, n_features)
print(f"Classes: {label_encoder.classes_}")  # ['Cooking', 'Eating', ...]

# 2. Train single model
model = train_xgboost_model(X, y)
print(f"Feature importance: {dict(zip(feature_names, model.feature_importances_))}")

# 3. Leave-one-out cross-validation
cv_results = leave_one_home_out_cv(lf_processed)
print(f"Mean accuracy: {cv_results['overall']['mean_accuracy']:.3f}")
print(f"Std accuracy: {cv_results['overall']['std_accuracy']:.3f}")

# Per-home results
for home, results in cv_results.items():
    if home != 'overall':
        print(f"{home}: {results['accuracy']:.3f} (n_test={results['n_test_samples']})")
```

**Why Leave-One-Out CV?** Tests the model's ability to generalize to completely new smart homes. Each home has different:
- Sensor configurations (some have Chair sensors, others don't)
- Resident behavior patterns
- Data collection periods

This is more realistic than random train/test splits within homes.

### 5. Persistence (`io/`)

#### `io/persistence.py`
Seven functions for saving/loading models and results:

```python
from ds_project.io.persistence import (
    save_processed_data, load_processed_data,
    save_model, load_model,
    save_cv_results, get_available_models
)

# Save processed dataset
lf_processed = preprocess_casas_features(load_all_casas_homes())
save_processed_data(lf_processed, "casas_preprocessed", subdir="processed")
# Saves to: data/processed/casas_preprocessed.parquet

# Save trained model with metadata
model = train_xgboost_model(X, y)
save_model(model, "xgb_baseline", metadata={
    "n_homes": len(get_available_homes()),
    "feature_count": len(feature_names),
    "accuracy": 0.85
})
# Saves to: models/xgb_baseline.pkl + models/xgb_baseline_metadata.json

# Save CV results
cv_results = leave_one_home_out_cv(lf_processed)
save_cv_results(cv_results, "baseline_experiment")
# Saves to: reports/baseline_experiment_YYYYMMDD_HHMMSS_cv_results.json

# Load back later
saved_model = load_model("xgb_baseline")
saved_data = load_processed_data("casas_preprocessed")
print(f"Available models: {get_available_models()}")
```

---

## Complete Function Reference

### Data Loading Functions

#### `load_casas_home(home_id, data_dir) -> LazyFrame`
**Purpose:** Load annotated features for a single CASAS home  
**Args:**
- `home_id` (str): Home identifier (e.g., 'csh101')
- `data_dir` (Path|str): Raw data directory path
**Returns:** LazyFrame with CASAS_FEATURES_SCHEMA + home_id column  
**Example:**
```python
lf = load_casas_home('csh101')
print(f"Loaded {lf.select(pl.count()).collect().item():,} samples")
```

#### `load_all_casas_homes(data_dir, home_range, sample_fraction) -> LazyFrame`
**Purpose:** Load and combine all CASAS homes  
**Args:**
- `home_range` (tuple): Start and end home numbers (default: (101, 130))
- `sample_fraction` (float|None): Fraction to sample from each home
**Returns:** Combined LazyFrame from all available homes  

#### `get_available_homes(data_dir) -> List[str]`
**Purpose:** Get list of homes with valid .ann.features.csv files  
**Returns:** List of home IDs like ['csh101', 'csh102', ...]

### Feature Transformation Functions

#### `apply_activity_grouping(lf) -> LazyFrame`
**Purpose:** Group 35 activities into 11 categories  
**Returns:** LazyFrame with activity_grouped and activity_original columns  

#### `drop_constant_features(lf) -> LazyFrame`
**Purpose:** Remove features with no variance (numDistinctSensors)  
**Returns:** LazyFrame with constant features removed  

#### `add_cyclic_time_features(lf) -> LazyFrame`  
**Purpose:** Add sine/cosine encoding for temporal features  
**Returns:** LazyFrame with hour_sin, hour_cos, dow_sin, dow_cos columns  

#### `preprocess_casas_features(lf, include_cyclic) -> LazyFrame`
**Purpose:** Complete preprocessing pipeline  
**Args:**
- `include_cyclic` (bool): Whether to add cyclic time features (default: True)
**Returns:** LazyFrame ready for model training  

### Model Training Functions

#### `prepare_training_data(lf, target_col, exclude_features) -> Tuple[ndarray, ndarray, List[str], LabelEncoder]`
**Purpose:** Convert LazyFrame to sklearn/XGBoost format  
**Returns:** (X, y, feature_names, label_encoder) tuple  

#### `train_xgboost_model(X, y, params) -> XGBClassifier`
**Purpose:** Train XGBoost with HAR-optimized parameters  
**Returns:** Trained XGBClassifier  

#### `leave_one_home_out_cv(lf, target_col) -> Dict[str, Dict[str, float]]`
**Purpose:** Perform leave-one-home-out cross-validation  
**Returns:** Dictionary with per-home results and overall statistics  

### I/O Functions

#### `save_processed_data(df, filename, subdir) -> Path`
**Purpose:** Save DataFrame to parquet format  
**Returns:** Path to saved file  

#### `load_processed_data(filename, subdir) -> LazyFrame`
**Purpose:** Load processed data from parquet  
**Returns:** LazyFrame with loaded data  

#### `save_model(model, model_name, metadata) -> Path`
**Purpose:** Save XGBoost model + metadata  
**Returns:** Path to model file  

#### `load_model(model_name) -> XGBClassifier`
**Purpose:** Load saved XGBoost model  

#### `save_cv_results(results, experiment_name, include_timestamp) -> Path`
**Purpose:** Save cross-validation results to JSON  

#### `get_available_models() -> List[str]`
**Purpose:** List available saved models  

---

## Data Schemas Deep Dive

### CASAS Features (37 total)

The CASAS dataset provides 37 engineered features per time window:

#### Temporal Features (3)
- `lastSensorEventHours`: Hour of day (0-23)
- `lastSensorEventSeconds`: Second of day (0-86399)  
- `lastSensorDayOfWeek`: Day of week (0-6)

#### Window Metadata (5)
- `windowDuration`: Length of sliding window
- `timeSinceLastSensorEvent`: Gap since previous sensor event
- `complexity`: Entropy measure of sensor activity
- `activityChange`: Whether activity changed in window
- `areaTransitions`: Number of location transitions

#### Sensor IDs (5)
- `prevDominantSensor1/2`: Most active sensors in previous windows
- `lastSensorID`: ID of most recent sensor
- `lastSensorLocation`: Location of most recent sensor
- `lastMotionLocation`: Location of motion sensor (-1 = no motion)

#### Sensor Count Features (11)
Weighted counts of sensor events per location:
- `sensorCount-{Bathroom, Bedroom, Chair, DiningRoom, Hall, Ignore, Kitchen, LivingRoom, Office, OutsideDoor, WorkArea}`

#### Elapsed Time Features (11)
Seconds since each location was last activated (max 86400 = 24 hours):
- `sensorElTime-{Bathroom, Bedroom, Chair, DiningRoom, Hall, Ignore, Kitchen, LivingRoom, Office, OutsideDoor, WorkArea}`

#### Special Cases
- `numDistinctSensors`: Always 0 (constant feature, dropped in preprocessing)
- `activity`: Target variable (35 unique activity labels)

**Missing Value Handling:**
- `lastMotionLocation = -1`: No motion in window (4.6% of data)
- `sensorElTime-* = 86400`: Sensor not present in home
- `sensorCount-* = 0`: Sensor not present in home

### Activity Grouping Schema

After preprocessing, activities are grouped from 35 to 11 classes:

```python
GROUPED_ACTIVITIES = {
    'Sleeping': ['Sleep', 'Sleep_Out_Of_Bed', 'Bed_Toilet_Transition'],
    'Cooking': ['Cook', 'Cook_Breakfast', 'Cook_Lunch', 'Cook_Dinner'], 
    'Eating': ['Eat', 'Eat_Breakfast', 'Eat_Lunch', 'Eat_Dinner'],
    'Washing_Dishes': ['Wash_Dishes', 'Wash_Breakfast_Dishes', 'Wash_Lunch_Dishes', 'Wash_Dinner_Dishes'],
    'Personal_Care': ['Bathe', 'Toilet', 'Personal_Hygiene', 'Groom', 'Dress'],
    'Medication': ['Morning_Meds', 'Evening_Meds', 'Take_Medicine'],
    'Leisure': ['Watch_TV', 'Read', 'Relax', 'Phone', 'Entertain_Guests'],
    'Work': ['Work_At_Table', 'Work_At_Desk'], 
    'Transitions': ['Enter_Home', 'Leave_Home', 'Step_Out'],
    'Drinking': ['Drink'],
    'Other': ['Other_Activity']
}
```

---

## Configuration Reference

The `Config` class provides 17 configurable parameters:

### Project Paths
- `project_root`: Path(".") - Project root directory
- `data_dir`: Path("data") - Data directory 
- `raw_data_dir`: Path("data/raw") - CASAS home directories
- `interim_data_dir`: Path("data/interim") - Intermediate processing
- `processed_data_dir`: Path("data/processed") - Final datasets
- `models_dir`: Path("models") - Saved model artifacts
- `reports_dir`: Path("reports") - CV results and figures
- `notebooks_dir`: Path("notebooks") - Jupyter notebooks

### Dataset Configuration
- `casas_home_range`: (101, 130) - Range of home IDs to process
- `casas_sample_fraction`: 0.2 - Fraction of data to use per home

### Activity Grouping
- `grouped_activities`: Dict mapping 11 groups to 35 original activities
- `features_to_drop`: ["numDistinctSensors"] - Features to remove

### Model Parameters
- `xgboost_params`: Dictionary of XGBoost hyperparameters
- `cv_strategy`: "leave_one_out" - Cross-validation method
- `cv_random_seed`: 42 - Reproducibility seed

### Preprocessing
- `include_cyclic_time`: True - Add sine/cosine time features
- `normalize_features`: False - XGBoost handles raw features well

### Compute Constraints
- `max_memory_gb`: 8.0 - Memory limit
- `n_jobs`: -1 - Use all CPU cores

**Environment Variable Override:**
Add `HAR_` prefix to override any setting:
```bash
export HAR_CASAS_SAMPLE_FRACTION=0.5
export HAR_XGBOOST_PARAMS__MAX_DEPTH=8  # Nested dict access with __
```

---

## Usage Examples

### Basic Pipeline Flow
```python
from ds_project.data.loaders import load_all_casas_homes
from ds_project.features.transforms import preprocess_casas_features  
from ds_project.models.training import prepare_training_data, train_xgboost_model
from ds_project.io.persistence import save_model

# 1. Load data
lf = load_all_casas_homes()
print(f"Loaded {lf.select(pl.count()).collect().item():,} samples")

# 2. Preprocess
lf_processed = preprocess_casas_features(lf)

# 3. Prepare for training
X, y, feature_names, label_encoder = prepare_training_data(lf_processed)

# 4. Train model
model = train_xgboost_model(X, y)

# 5. Save results
save_model(model, "baseline_model", metadata={
    "features": feature_names,
    "classes": label_encoder.classes_.tolist()
})
```

### Single Home Analysis
```python
from ds_project.data.loaders import load_casas_home
from ds_project.features.transforms import preprocess_casas_features

# Load and analyze single home
lf = load_casas_home('csh101')
lf_processed = preprocess_casas_features(lf)

# Collect small sample for inspection
sample = lf_processed.head(100).collect()

# Activity distribution
activity_counts = sample.group_by('activity_grouped').count()
print("Activity distribution:")
print(activity_counts.sort('count', descending=True))

# Feature statistics
numeric_features = [col for col in sample.columns if sample[col].dtype.is_numeric()]
feature_stats = sample.select(numeric_features).describe()
print("Feature statistics:")
print(feature_stats)
```

### Leave-One-Out Cross-Validation
```python
from ds_project.models.training import leave_one_home_out_cv
from ds_project.io.persistence import save_cv_results

# Load and preprocess all data
lf = load_all_casas_homes()
lf_processed = preprocess_casas_features(lf)

# Run leave-one-out CV (this takes time!)
print("Starting leave-one-out CV across 30 homes...")
cv_results = leave_one_home_out_cv(lf_processed)

# Print results
print(f"Mean accuracy: {cv_results['overall']['mean_accuracy']:.3f}")
print(f"Std accuracy: {cv_results['overall']['std_accuracy']:.3f}")
print(f"Range: {cv_results['overall']['min_accuracy']:.3f} - {cv_results['overall']['max_accuracy']:.3f}")

# Save detailed results
save_cv_results(cv_results, "baseline_experiment")

# Find hardest homes to predict
home_results = [(home, results['accuracy']) 
                for home, results in cv_results.items() 
                if home != 'overall']
home_results.sort(key=lambda x: x[1])  # Sort by accuracy

print("Hardest homes:")
for home, acc in home_results[:5]:
    print(f"  {home}: {acc:.3f}")

print("Easiest homes:")
for home, acc in home_results[-5:]:
    print(f"  {home}: {acc:.3f}")
```

### Model Persistence Workflow
```python
from ds_project.io.persistence import save_model, load_model, get_available_models

# After training a model...
model = train_xgboost_model(X, y)

# Save with rich metadata
save_model(model, "experiment_001", metadata={
    "experiment": "baseline XGBoost",
    "n_samples": len(X),
    "n_features": len(feature_names),
    "feature_names": feature_names,
    "classes": label_encoder.classes_.tolist(),
    "sample_fraction": 0.2,
    "include_cyclic": True
})

# Later, load and use
available_models = get_available_models()
print(f"Available models: {available_models}")

loaded_model = load_model("experiment_001")
# Model is ready to use for predictions
```

---

## Key Design Decisions Explained

### Why LazyFrame-First Processing?

The CASAS dataset contains ~14M annotated samples. Loading all data into memory would require significant RAM. Polars LazyFrame allows us to:

1. **Build transformation pipelines** without immediate execution
2. **Optimize query plans** automatically (predicate pushdown, projection pruning)
3. **Stream processing** - only materialize data when needed (`.collect()`)
4. **Memory efficiency** - process larger-than-memory datasets

```python
# This builds a query plan, doesn't load data
lf = (
    load_all_casas_homes()
    .pipe(preprocess_casas_features)
    .filter(pl.col('activity_grouped') == 'Cooking')
    .select(['sensorCount-Kitchen', 'activity_grouped'])
)

# Only now does data processing happen
cooking_data = lf.collect()  # Much smaller result set
```

### Why Activity Grouping (35 → 11)?

The original CASAS dataset has severe class imbalance:
- `Other_Activity`: 28-44% of all samples
- Many activities: <1% each (e.g., `Entertain_Guests`, `Take_Medicine`)

Class imbalance creates several problems:
1. **Model bias** toward dominant classes
2. **Poor performance** on rare but important activities  
3. **Unstable training** with very few positive examples

Grouping similar activities creates:
1. **More balanced classes** with sufficient training examples
2. **Semantic coherence** - grouped activities share similar sensor patterns
3. **Practical utility** - 11 categories are more interpretable than 35

The grouping preserves the original labels as `activity_original` for analysis.

### Why XGBoost for HAR?

XGBoost is well-suited for this HAR task because:

1. **Handles sparse features** naturally (many sensor locations absent in homes)
2. **No preprocessing required** for missing values (treats -1 and 86400 as valid)
3. **Built-in class weighting** for imbalanced datasets
4. **Feature importance** interpretability for understanding sensor contributions
5. **Robust to outliers** common in sensor data
6. **Fast training** suitable for 30-fold leave-one-out CV

The configured parameters balance accuracy and overfitting:
- `max_depth: 6` - Prevents overfitting to individual home quirks
- `subsample: 0.8` - Row sampling for robustness
- `colsample_bytree: 0.8` - Feature sampling for generalization

---

## Development Workflow

### Code Quality Pipeline
```bash
# Format code
ruff format src/

# Lint code  
ruff check src/

# Type check
mypy src/

# All checks pass? Commit changes
git add . && git commit -m "feat: implement new feature"
```

### Typical Development Session
```python
# 1. Start with exploration
from ds_project.config import config
from ds_project.data.loaders import get_available_homes, load_casas_home

homes = get_available_homes()
lf = load_casas_home(homes[0])

# 2. Experiment with transformations
from ds_project.features.transforms import preprocess_casas_features
lf_processed = preprocess_casas_features(lf)

# 3. Quick training test
from ds_project.models.training import prepare_training_data, train_xgboost_model
X, y, features, encoder = prepare_training_data(lf_processed.head(1000))
model = train_xgboost_model(X, y)

# 4. When satisfied, scale up and save
# ... full pipeline with all data and save results
```

---

This comprehensive overview documents all 775 lines of code across 17 functions in 5 modules. The codebase implements a complete, production-ready pipeline for Human Activity Recognition using modern Python data science practices.