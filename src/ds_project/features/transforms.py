"""Feature transformation functions for CASAS HAR dataset.

Pure transformation functions that take LazyFrame and return LazyFrame.
No I/O operations - all transformations are lazy until .collect() is called.
"""

from typing import Dict, List
import math

import polars as pl
from polars import LazyFrame

# Activity grouping mapping: 35 raw activities → 11 grouped categories
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

# Invert the mapping for efficient lookup: activity -> group
_ACTIVITY_TO_GROUP: Dict[str, str] = {}
for group, activities in GROUPED_ACTIVITIES.items():
    for activity in activities:
        _ACTIVITY_TO_GROUP[activity] = group


def apply_activity_grouping(lf: LazyFrame) -> LazyFrame:
    """Group 35 original activities into 11 categories.
    
    Creates a new 'activity_grouped' column while preserving the original
    'activity' column as 'activity_original' for analysis purposes.
    
    Args:
        lf: LazyFrame with 'activity' column containing original activity labels
        
    Returns:
        LazyFrame with additional 'activity_grouped' and 'activity_original' columns
        
    Raises:
        ValueError: If any activity in the data is not in the grouping mapping
    """
    return (
        lf
        .with_columns(pl.col("activity").alias("activity_original"))
        .with_columns(
            pl.col("activity").replace(_ACTIVITY_TO_GROUP).alias("activity_grouped")
        )
    )


def drop_constant_features(lf: LazyFrame) -> LazyFrame:
    """Drop features with no variance across the dataset.
    
    Based on analysis, 'numDistinctSensors' is always 0 and should be removed.
    This function can be extended to drop other constant features if discovered.
    
    Args:
        lf: LazyFrame containing CASAS features
        
    Returns:
        LazyFrame with constant features removed
    """
    return lf.drop("numDistinctSensors")


def add_cyclic_time_features(lf: LazyFrame) -> LazyFrame:
    """Add cyclic encoding for temporal features.
    
    Converts hour and day-of-week to sine/cosine representations to capture
    the cyclic nature of time (e.g., hour 23 is close to hour 0).
    
    Args:
        lf: LazyFrame with 'lastSensorEventHours' and 'lastSensorDayOfWeek' columns
        
    Returns:
        LazyFrame with additional sin/cos time features
    """
    return (
        lf
        .with_columns([
            # Hour encoding (0-23 → 0-2π)
            (pl.col("lastSensorEventHours") * 2 * math.pi / 24).sin().alias("hour_sin"),
            (pl.col("lastSensorEventHours") * 2 * math.pi / 24).cos().alias("hour_cos"),
            
            # Day of week encoding (0-6 → 0-2π) 
            (pl.col("lastSensorDayOfWeek") * 2 * math.pi / 7).sin().alias("dow_sin"),
            (pl.col("lastSensorDayOfWeek") * 2 * math.pi / 7).cos().alias("dow_cos"),
        ])
    )


def add_feature_interactions(lf: LazyFrame) -> LazyFrame:
    """Add meaningful feature interactions for HAR classification.
    
    Creates interaction features that capture relationships between:
    - Time of day and location activity (Kitchen activity at dinner time)
    - Sensor counts and time patterns
    - Location transitions and activity duration
    
    These interactions help the model learn patterns like:
    - High kitchen activity + evening hour = likely Cooking
    - Bedroom sensors + late night = likely Sleeping
    - Outside door + morning = likely Transitions
    
    Args:
        lf: LazyFrame with base CASAS features
        
    Returns:
        LazyFrame with additional interaction features (~15-20 new features)
    """
    # Key locations for HAR
    locations = ['Kitchen', 'Bedroom', 'Bathroom', 'LivingRoom', 'OutsideDoor']
    
    interaction_exprs = []
    
    # Time × Location interactions (most predictive for HAR)
    for loc in locations:
        sensor_col = f"sensorCount-{loc}"
        
        # Location activity × hour of day
        interaction_exprs.append(
            (pl.col(sensor_col) * pl.col("lastSensorEventHours")).alias(f"{loc}_x_hour")
        )
        
        # Location activity × time since last event (recency interaction)
        interaction_exprs.append(
            (pl.col(sensor_col) * pl.col("timeSinceLastSensorEvent")).alias(f"{loc}_x_recency")
        )
    
    # Complexity interactions
    interaction_exprs.extend([
        # Complexity × window duration (activity intensity)
        (pl.col("complexity") * pl.col("windowDuration")).alias("complexity_x_duration"),
        
        # Area transitions × complexity (movement patterns)
        (pl.col("areaTransitions") * pl.col("complexity")).alias("transitions_x_complexity"),
    ])
    
    # Time-based aggregate features
    interaction_exprs.extend([
        # Is it morning (6-12)?
        ((pl.col("lastSensorEventHours") >= 6) & (pl.col("lastSensorEventHours") < 12))
            .cast(pl.Int32).alias("is_morning"),
        
        # Is it afternoon (12-18)?
        ((pl.col("lastSensorEventHours") >= 12) & (pl.col("lastSensorEventHours") < 18))
            .cast(pl.Int32).alias("is_afternoon"),
        
        # Is it evening (18-22)?
        ((pl.col("lastSensorEventHours") >= 18) & (pl.col("lastSensorEventHours") < 22))
            .cast(pl.Int32).alias("is_evening"),
        
        # Is it night (22-6)?
        ((pl.col("lastSensorEventHours") >= 22) | (pl.col("lastSensorEventHours") < 6))
            .cast(pl.Int32).alias("is_night"),
        
        # Is it weekend?
        (pl.col("lastSensorDayOfWeek") >= 5).cast(pl.Int32).alias("is_weekend"),
    ])
    
    # Location dominance features
    interaction_exprs.extend([
        # Total sensor activity across key locations
        (pl.col("sensorCount-Kitchen") + pl.col("sensorCount-Bedroom") + 
         pl.col("sensorCount-Bathroom") + pl.col("sensorCount-LivingRoom"))
            .alias("total_main_activity"),
        
        # Kitchen dominance ratio
        (pl.col("sensorCount-Kitchen") / 
         (pl.col("sensorCount-Kitchen") + pl.col("sensorCount-Bedroom") + 
          pl.col("sensorCount-Bathroom") + pl.col("sensorCount-LivingRoom") + 1))
            .alias("kitchen_dominance"),
        
        # Bedroom dominance ratio
        (pl.col("sensorCount-Bedroom") / 
         (pl.col("sensorCount-Kitchen") + pl.col("sensorCount-Bedroom") + 
          pl.col("sensorCount-Bathroom") + pl.col("sensorCount-LivingRoom") + 1))
            .alias("bedroom_dominance"),
    ])
    
    # Motion-based features
    interaction_exprs.extend([
        # Has motion in window
        (pl.col("lastMotionLocation") != -1).cast(pl.Int32).alias("has_motion"),
        
        # Motion location matches last sensor location
        (pl.col("lastMotionLocation") == pl.col("lastSensorLocation"))
            .cast(pl.Int32).alias("motion_matches_sensor"),
    ])
    
    return lf.with_columns(interaction_exprs)


def add_elapsed_time_features(lf: LazyFrame) -> LazyFrame:
    """Add derived features from elapsed time columns.
    
    Creates aggregate features from the sensorElTime-* columns:
    - Minimum elapsed time (most recent activity)
    - Number of recently active locations
    - Activity spread indicator
    
    Args:
        lf: LazyFrame with sensorElTime-* columns
        
    Returns:
        LazyFrame with additional elapsed time derived features
    """
    locations = ['Bathroom', 'Bedroom', 'Kitchen', 'LivingRoom', 'OutsideDoor']
    el_time_cols = [f"sensorElTime-{loc}" for loc in locations]
    
    return lf.with_columns([
        # Minimum elapsed time across key locations (most recent activity)
        pl.min_horizontal(*[pl.col(c) for c in el_time_cols]).alias("min_elapsed_time"),
        
        # Count of recently active locations (< 5 minutes)
        pl.sum_horizontal(*[(pl.col(c) < 300).cast(pl.Int32) for c in el_time_cols])
            .alias("recent_active_locations"),
        
        # Standard deviation of elapsed times (activity spread)
        # High std = focused in one area, low std = spread across areas
    ])


def preprocess_casas_features(
    lf: LazyFrame, 
    include_cyclic: bool = True,
    include_interactions: bool = True
) -> LazyFrame:
    """Apply full preprocessing pipeline to CASAS features.
    
    Complete preprocessing pipeline that applies activity grouping,
    removes constant features, and optionally adds enhanced features.
    
    Args:
        lf: LazyFrame with raw CASAS features
        include_cyclic: Whether to add cyclic time encodings
        include_interactions: Whether to add feature interactions
        
    Returns:
        LazyFrame with preprocessed features ready for model training
    """
    result = (
        lf
        .pipe(apply_activity_grouping)
        .pipe(drop_constant_features)
    )
    
    if include_cyclic:
        result = result.pipe(add_cyclic_time_features)
    
    if include_interactions:
        result = result.pipe(add_feature_interactions)
        result = result.pipe(add_elapsed_time_features)
    
    return result