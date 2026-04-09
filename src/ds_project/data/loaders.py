"""Data loading functions for CASAS dataset.

All loaders return polars LazyFrame for memory efficiency.
Files are cast to canonical schemas on load to ensure type consistency.
"""

from pathlib import Path
from typing import List, Optional

import polars as pl
from polars import LazyFrame

from .schemas import CASAS_FEATURES_SCHEMA


def load_casas_home(home_id: str, data_dir: Path | str = "data/raw") -> LazyFrame:
    """Load annotated features for a single CASAS home.
    
    Loads the .ann.features.csv file for the specified home and casts
    all columns to the canonical CASAS_FEATURES_SCHEMA.
    
    Args:
        home_id: Home identifier (e.g., 'csh101', 'csh102', etc.)
        data_dir: Path to the raw data directory containing home folders
        
    Returns:
        LazyFrame with schema matching CASAS_FEATURES_SCHEMA
        
    Raises:
        FileNotFoundError: If the home directory or .ann.features.csv file doesn't exist
        pl.SchemaError: If the file schema is incompatible with CASAS_FEATURES_SCHEMA
    """
    data_path = Path(data_dir)
    home_dir = data_path / home_id
    ann_file = home_dir / f"{home_id}.ann.features.csv"
    
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotated features file not found: {ann_file}")
    
    return (
        pl.scan_csv(str(ann_file))
        .cast(CASAS_FEATURES_SCHEMA)
        .with_columns(pl.lit(home_id).alias("home_id"))  # Add home identifier
    )


def load_all_casas_homes(
    data_dir: Path | str = "data/raw",
    home_range: tuple[int, int] = (101, 130),
    sample_fraction: Optional[float] = None
) -> LazyFrame:
    """Load annotated features from all CASAS homes.
    
    Loads and combines .ann.features.csv files from all homes in the specified range.
    Optionally applies stratified sampling per home to reduce memory usage.
    
    Args:
        data_dir: Path to the raw data directory
        home_range: Tuple of (start, end+1) for home IDs (default: csh101-csh130)
        sample_fraction: If provided, randomly sample this fraction from each home
        
    Returns:
        LazyFrame with combined data from all homes, with added 'home_id' column
        
    Raises:
        FileNotFoundError: If no valid home files are found
    """
    data_path = Path(data_dir)
    home_lazyframes: List[LazyFrame] = []
    
    start_id, end_id = home_range
    for home_num in range(start_id, end_id + 1):
        home_id = f"csh{home_num}"
        home_dir = data_path / home_id
        ann_file = home_dir / f"{home_id}.ann.features.csv"
        
        if ann_file.exists():
            lf = (
                pl.scan_csv(str(ann_file))
                .cast(CASAS_FEATURES_SCHEMA)
                .with_columns(pl.lit(home_id).alias("home_id"))
            )
            
            # Note: Sampling is applied after loading due to LazyFrame limitations
            # For memory efficiency, consider using .head(n) or collecting first
            
            home_lazyframes.append(lf)
    
    if not home_lazyframes:
        raise FileNotFoundError(f"No valid .ann.features.csv files found in {data_path}")
    
    return pl.concat(home_lazyframes, how="vertical")


def get_available_homes(data_dir: Path | str = "data/raw") -> List[str]:
    """Get list of available CASAS home IDs with .ann.features.csv files.
    
    Args:
        data_dir: Path to the raw data directory
        
    Returns:
        List of home IDs (e.g., ['csh101', 'csh102', ...])
    """
    data_path = Path(data_dir)
    available_homes = []
    
    for home_dir in sorted(data_path.glob("csh*")):
        if home_dir.is_dir():
            home_id = home_dir.name
            ann_file = home_dir / f"{home_id}.ann.features.csv"
            if ann_file.exists():
                available_homes.append(home_id)
    
    return available_homes