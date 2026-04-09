"""Schema definitions for CASAS Human Activity Recognition dataset.

All data schemas are defined here as polars.type_aliases.SchemaDefinition dicts.
This ensures consistent typing across all data loading and processing operations.
"""

import polars as pl
from polars.type_aliases import SchemaDefinition

# Schema for the annotated feature files (.ann.features.csv)
# Based on analysis of 37 features from CASAS dataset
CASAS_FEATURES_SCHEMA: SchemaDefinition = {
    # Temporal features (3)
    "lastSensorEventHours": pl.Int32,       # 0-23
    "lastSensorEventSeconds": pl.Int32,     # 0-86399  
    "lastSensorDayOfWeek": pl.Int32,        # 0-6
    
    # Window metadata (5)
    "windowDuration": pl.Float64,
    "timeSinceLastSensorEvent": pl.Float64,
    "complexity": pl.Float64,               # entropy measure
    "activityChange": pl.Float64,
    "areaTransitions": pl.Float64,
    
    # Sensor IDs (5) - raw identifiers, may contain -1 for "none"
    "prevDominantSensor1": pl.Int32,
    "prevDominantSensor2": pl.Int32, 
    "lastSensorID": pl.Int32,
    "lastSensorLocation": pl.Int32,
    "lastMotionLocation": pl.Int32,         # -1 means "no motion in window"
    
    # Sensor count features (11) - weighted counts per location
    "sensorCount-Bathroom": pl.Float64,
    "sensorCount-Bedroom": pl.Float64,
    "sensorCount-Chair": pl.Float64,
    "sensorCount-DiningRoom": pl.Float64,
    "sensorCount-Hall": pl.Float64,
    "sensorCount-Ignore": pl.Float64,
    "sensorCount-Kitchen": pl.Float64,
    "sensorCount-LivingRoom": pl.Float64,
    "sensorCount-Office": pl.Float64,
    "sensorCount-OutsideDoor": pl.Float64,
    "sensorCount-WorkArea": pl.Float64,
    
    # Elapsed time features (11) - seconds since last activation (max 86400)
    "sensorElTime-Bathroom": pl.Float64,
    "sensorElTime-Bedroom": pl.Float64,
    "sensorElTime-Chair": pl.Float64,
    "sensorElTime-DiningRoom": pl.Float64,
    "sensorElTime-Hall": pl.Float64,
    "sensorElTime-Ignore": pl.Float64,
    "sensorElTime-Kitchen": pl.Float64,
    "sensorElTime-LivingRoom": pl.Float64,
    "sensorElTime-Office": pl.Float64,
    "sensorElTime-OutsideDoor": pl.Float64,
    "sensorElTime-WorkArea": pl.Float64,
    
    # Always zero - will be dropped in preprocessing
    "numDistinctSensors": pl.Int32,
    
    # Target variable - activity class (35 unique values)
    "activity": pl.Utf8,
}

# Grouped activities schema after feature engineering (35 → 11 classes)
GROUPED_ACTIVITIES_SCHEMA: SchemaDefinition = {
    **{k: v for k, v in CASAS_FEATURES_SCHEMA.items() if k != "activity"},
    "activity_grouped": pl.Utf8,            # 11 grouped categories
    "activity_original": pl.Utf8,           # preserve original for analysis
}