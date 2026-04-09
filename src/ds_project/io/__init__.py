"""File I/O operations for models, results, and processed data.

Handles reading/writing of model artifacts (.pkl, .json), processed datasets
(parquet), and evaluation results. The only module allowed to write to
models/, reports/, and data/processed/ directories.
"""