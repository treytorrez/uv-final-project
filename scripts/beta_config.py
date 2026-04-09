"""Beta baseline model configuration for overnight training.

This configuration is optimized for:
- Extended overnight training with early stopping
- Lower learning rate for more iterations  
- Rich logging and checkpointing
- Full leave-one-out CV across 30 homes
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


@dataclass
class BetaConfig:
    """Configuration for beta baseline overnight training."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT METADATA
    # ═══════════════════════════════════════════════════════════════════════════
    experiment_name: str = "beta_baseline"
    experiment_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    description: str = "Beta baseline with early stopping, cyclic features, and feature interactions"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    data_dir: Path = Path("data/raw")
    home_range: tuple = (101, 130)  # csh101 to csh130
    sample_fraction: float = 0.2    # 20% sample for overnight run
    random_seed: int = 42
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════════
    include_cyclic_time: bool = True           # sin/cos for hour and day-of-week
    include_feature_interactions: bool = True  # Kitchen×Hour, etc.
    drop_constant_features: bool = True        # Remove numDistinctSensors
    
    # Features to drop (constant or problematic)
    features_to_drop: List[str] = field(default_factory=lambda: ["numDistinctSensors"])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # XGBOOST PARAMETERS (OVERNIGHT OPTIMIZED)
    # ═══════════════════════════════════════════════════════════════════════════
    # Lower learning rate + more trees + early stopping = better overnight results
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        # Core parameters
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',          # Faster training for large datasets
        
        # Tree structure
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,                     # Minimum loss reduction for split
        
        # Learning rate (low for overnight refinement)
        'learning_rate': 0.01,          # Lower LR = more trees = better generalization
        'n_estimators': 1500,           # High limit, early stopping will kick in
        
        # Regularization
        'subsample': 0.8,               # Row sampling
        'colsample_bytree': 0.8,        # Feature sampling per tree
        'colsample_bylevel': 0.8,       # Feature sampling per level
        'reg_alpha': 0.1,               # L1 regularization
        'reg_lambda': 1.0,              # L2 regularization
        
        # Reproducibility
        'random_state': 42,
        'n_jobs': -1,                   # Use all cores
        
        # Verbosity
        'verbosity': 1,
    })
    
    # Early stopping configuration
    early_stopping_rounds: int = 100    # Stop if no improvement for 100 rounds
    eval_fraction: float = 0.1          # Hold out 10% of training for early stopping eval
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    cv_strategy: str = "leave_one_out"  # Train on 29 homes, test on 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OUTPUT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    output_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    checkpoint_dir: Path = Path("models/checkpoints")
    
    # What to save
    save_models: bool = True            # Save trained model for each fold
    save_predictions: bool = True       # Save predictions for analysis
    save_feature_importance: bool = True
    save_confusion_matrices: bool = True
    save_training_logs: bool = True
    checkpoint_interval: int = 5        # Save checkpoint every N homes
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════════════════
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("reports/beta_training.log"))
    progress_interval: int = 1          # Log progress every N homes
    
    @property
    def experiment_id(self) -> str:
        """Unique identifier for this experiment run."""
        return f"{self.experiment_name}_{self.experiment_timestamp}"
    
    @property
    def results_path(self) -> Path:
        """Path to save results for this experiment."""
        return self.output_dir / self.experiment_id
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVITY GROUPING (35 → 11 classes)
# ═══════════════════════════════════════════════════════════════════════════════
GROUPED_ACTIVITIES: Dict[str, List[str]] = {
    'Sleeping': ['Sleep', 'Sleep_Out_Of_Bed', 'Bed_Toilet_Transition'],
    'Cooking': ['Cook', 'Cook_Breakfast', 'Cook_Lunch', 'Cook_Dinner'],
    'Eating': ['Eat', 'Eat_Breakfast', 'Eat_Lunch', 'Eat_Dinner'],
    'Washing_Dishes': ['Wash_Dishes', 'Wash_Breakfast_Dishes', 
                       'Wash_Lunch_Dishes', 'Wash_Dinner_Dishes'],
    'Personal_Care': ['Bathe', 'Toilet', 'Personal_Hygiene', 'Groom', 'Dress'],
    'Medication': ['Morning_Meds', 'Evening_Meds', 'Take_Medicine'],
    'Leisure': ['Watch_TV', 'Read', 'Relax', 'Phone', 'Entertain_Guests'],
    'Work': ['Work_At_Table', 'Work_At_Desk'],
    'Transitions': ['Enter_Home', 'Leave_Home', 'Step_Out'],
    'Drinking': ['Drink'],
    'Other': ['Other_Activity']
}

# Inverted mapping for efficient lookup
ACTIVITY_TO_GROUP: Dict[str, str] = {}
for group, activities in GROUPED_ACTIVITIES.items():
    for activity in activities:
        ACTIVITY_TO_GROUP[activity] = group


# Default config instance
beta_config = BetaConfig()
