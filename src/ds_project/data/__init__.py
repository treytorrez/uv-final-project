"""Data loading, schema definitions, and validation for CASAS dataset.

This module handles all I/O operations for the CASAS smart home sensor data.
Loaders return polars LazyFrame objects for efficient memory usage.
Schemas are defined once and used consistently across the pipeline.
"""