"""Human Activity Recognition project using CASAS smart home dataset.

This package implements a machine learning pipeline for classifying daily activities
from ambient sensor data across 30 smart homes, with a focus on generalization
via leave-one-out cross-validation.

Dataset: UCI Human Activity Recognition from Continuous Ambient Sensor Data
Primary Model: XGBoost for handling sparse features and class imbalance
Validation: Leave-one-out CV across homes (train on 29, test on 1)
Activity Grouping: 35 raw activities → 11 grouped categories
"""

__version__ = "0.1.0"