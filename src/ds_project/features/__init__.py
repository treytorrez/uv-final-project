"""Feature engineering and transformation pipeline.

Pure transformation functions that take LazyFrame and return LazyFrame.
No I/O operations - all file operations should go through data/ or io/ modules.
Includes activity grouping (35 → 11 classes) and sensor feature processing.
"""