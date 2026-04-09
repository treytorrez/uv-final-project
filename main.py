#!/usr/bin/env python3
"""
CASAS Human Activity Recognition - Main Entry Point

Quick demo of the HAR pipeline for activity classification.
Run this after entering `nix develop` to test the environment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    try:
        from ds_project.data.loaders import get_available_homes, load_casas_home
        from ds_project.features.transforms import preprocess_casas_features
        from ds_project.config import config
        
        print("🏠 CASAS Human Activity Recognition Pipeline")
        print("=" * 50)
        
        # Check available data
        print("\n1. Checking available CASAS homes...")
        available_homes = get_available_homes()
        print(f"   Found {len(available_homes)} homes: {available_homes[:3]}...")
        
        if not available_homes:
            print("   ❌ No CASAS home data found in data/raw/")
            print("   Make sure the CASAS dataset is extracted to data/raw/")
            sys.exit(1)
        
        # Load sample data
        print(f"\n2. Loading sample data from {available_homes[0]}...")
        lf = load_casas_home(available_homes[0])
        sample_size = lf.select("activity").collect().height
        print(f"   📊 Loaded {sample_size:,} activity samples")
        
        # Show original activities
        activities = lf.select("activity").unique().collect()["activity"].to_list()
        print(f"   🎯 Found {len(activities)} unique activities")
        
        # Apply preprocessing
        print("\n3. Applying preprocessing pipeline...")
        lf_processed = preprocess_casas_features(lf)
        
        # Show grouped activities  
        grouped_activities = lf_processed.select("activity_grouped").unique().collect()["activity_grouped"].to_list()
        print(f"   ✅ Grouped into {len(grouped_activities)} categories: {sorted(grouped_activities)}")
        
        # Show feature info
        original_features = len(lf.schema)
        processed_features = len(lf_processed.schema)
        print(f"   🔧 Features: {original_features} → {processed_features} (added cyclic time features)")
        
        print("\n4. Configuration:")
        print(f"   Sample fraction: {config.casas_sample_fraction}")
        print(f"   XGBoost params: {list(config.xgboost_params.keys())}")
        print(f"   CV strategy: {config.cv_strategy}")
        
        print("\n✅ Pipeline test successful!")
        print("\nNext steps:")
        print("- Run `jupyter lab` to explore notebooks/")
        print("- Use `ds_project.models.training` for leave-one-out CV")
        print("- Check `ds_project.io.persistence` for saving results")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're in the nix development shell:")
        print("  nix develop")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()