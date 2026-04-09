"""File I/O operations for models, results, and processed data.

Handles reading/writing of model artifacts, processed datasets, and evaluation results.
The only module allowed to write to models/, reports/, and data/processed/ directories.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import pickle
from datetime import datetime

import polars as pl
from polars import LazyFrame, DataFrame
import xgboost as xgb

from ..config import config


def save_processed_data(
    df: DataFrame | LazyFrame, 
    filename: str,
    subdir: str = "processed"
) -> Path:
    """Save processed DataFrame to parquet format.
    
    Args:
        df: Polars DataFrame or LazyFrame to save
        filename: Output filename (without extension)
        subdir: Subdirectory within data/ (processed, interim)
        
    Returns:
        Path to saved file
    """
    output_dir = config.data_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{filename}.parquet"
    
    if isinstance(df, LazyFrame):
        df = df.collect()
    
    df.write_parquet(output_path)
    print(f"Saved processed data: {output_path}")
    
    return output_path


def load_processed_data(filename: str, subdir: str = "processed") -> LazyFrame:
    """Load processed data from parquet format.
    
    Args:
        filename: Filename (without extension)
        subdir: Subdirectory within data/
        
    Returns:
        LazyFrame with the loaded data
    """
    file_path = config.data_dir / subdir / f"{filename}.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    return pl.scan_parquet(file_path)


def save_model(
    model: xgb.XGBClassifier,
    model_name: str,
    metadata: Dict[str, Any] | None = None
) -> Path:
    """Save trained XGBoost model and metadata.
    
    Args:
        model: Trained XGBoost classifier
        model_name: Model identifier for filename
        metadata: Additional metadata to save alongside model
        
    Returns:
        Path to saved model file
    """
    output_dir = config.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        "model_type": "XGBClassifier",
        "saved_at": datetime.now().isoformat(),
        "model_params": model.get_params(),
    })
    
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    
    return model_path


def load_model(model_name: str) -> xgb.XGBClassifier:
    """Load saved XGBoost model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Loaded XGBoost classifier
    """
    model_path = config.models_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model


def save_cv_results(
    results: Dict[str, Any],
    experiment_name: str,
    include_timestamp: bool = True
) -> Path:
    """Save cross-validation results to JSON.
    
    Args:
        results: Results dictionary from leave_one_home_out_cv()
        experiment_name: Experiment identifier
        include_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved results file
    """
    output_dir = config.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = experiment_name
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}"
    
    output_path = output_dir / f"{filename}_cv_results.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, "item"):
            return obj.item()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        else:
            return str(obj)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"Saved CV results: {output_path}")
    return output_path


def get_available_models() -> List[str]:
    """Get list of available saved models.
    
    Returns:
        List of model names (without .pkl extension)
    """
    models_dir = config.models_dir
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pkl"))
    return [f.stem for f in model_files]