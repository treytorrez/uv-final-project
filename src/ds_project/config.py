"""Configuration settings for CASAS HAR project.

All configurable paths, hyperparameters, and environment variable bindings
are centralized here using pydantic-settings for type safety and validation.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for CASAS Human Activity Recognition project."""
    
    model_config = SettingsConfigDict(
        env_prefix="HAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Project paths
    project_root: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    interim_data_dir: Path = Field(default=Path("data/interim"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models"))
    reports_dir: Path = Field(default=Path("reports"))
    notebooks_dir: Path = Field(default=Path("notebooks"))
    
    # CASAS dataset configuration
    casas_home_range: Tuple[int, int] = Field(default=(101, 130))  # csh101-csh130
    casas_sample_fraction: float = Field(default=0.2)  # 20% sample for local compute
    
    # Activity grouping (35 → 11 classes)
    grouped_activities: Dict[str, List[str]] = Field(default={
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
    })
    
    # Features to drop (constant or irrelevant)
    features_to_drop: List[str] = Field(default=["numDistinctSensors"])
    
    # Model hyperparameters
    xgboost_params: Dict = Field(default={
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 100,
    })
    
    # Cross-validation settings
    cv_strategy: str = Field(default="leave_one_out")  # Leave-one-home-out
    cv_random_seed: int = Field(default=42)
    
    # Preprocessing options
    include_cyclic_time: bool = Field(default=True)
    normalize_features: bool = Field(default=False)  # XGBoost handles raw features well
    
    # Compute constraints
    max_memory_gb: float = Field(default=8.0)
    n_jobs: int = Field(default=-1)  # Use all available cores
    
    @property
    def available_homes(self) -> List[str]:
        """Generate list of expected home IDs based on configured range."""
        start, end = self.casas_home_range
        return [f"csh{i}" for i in range(start, end + 1)]
    
    @property
    def activity_to_group_mapping(self) -> Dict[str, str]:
        """Invert the grouped_activities mapping for efficient lookup."""
        mapping = {}
        for group, activities in self.grouped_activities.items():
            for activity in activities:
                mapping[activity] = group
        return mapping


# Global configuration instance
config = Config()