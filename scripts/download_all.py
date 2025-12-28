"""
Master Download Script for Drug Discovery Datasets

Downloads all datasets from:
- ChEMBL
- DrugBank/PubChem
- FDA
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from download_chembl import main as download_chembl
from download_drugbank import main as download_drugbank
from download_fda import main as download_fda
from prepare_dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download all drug discovery datasets"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test mode - download small samples only"
    )
    parser.add_argument(
        "--skip-chembl",
        action="store_true",
        help="Skip ChEMBL download"
    )
    parser.add_argument(
        "--skip-drugbank",
        action="store_true",
        help="Skip DrugBank download"
    )
    parser.add_argument(
        "--skip-fda",
        action="store_true",
        help="Skip FDA download"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Skip downloads, only prepare dataset"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Drug Discovery Dataset Downloader")
    print("="*60)
    
    if not args.prepare_only:
        # Download ChEMBL data
        if not args.skip_chembl:
            print("\n" + "-"*40)
            print("STEP 1: Downloading ChEMBL Data")
            print("-"*40)
            try:
                download_chembl(test_mode=args.test)
            except Exception as e:
                print(f"Warning: ChEMBL download failed: {e}")
        
        # Download DrugBank data
        if not args.skip_drugbank:
            print("\n" + "-"*40)
            print("STEP 2: Downloading DrugBank/PubChem Data")
            print("-"*40)
            try:
                download_drugbank(test_mode=args.test)
            except Exception as e:
                print(f"Warning: DrugBank download failed: {e}")
        
        # Download FDA data
        if not args.skip_fda:
            print("\n" + "-"*40)
            print("STEP 3: Downloading FDA Data")
            print("-"*40)
            try:
                download_fda(test_mode=args.test)
            except Exception as e:
                print(f"Warning: FDA download failed: {e}")
    
    # Prepare combined dataset
    print("\n" + "-"*40)
    print("STEP 4: Preparing Combined Dataset")
    print("-"*40)
    try:
        prepare_dataset(augment=True, test_mode=args.test)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise
    
    print("\n" + "="*60)
    print("ALL DOWNLOADS AND PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review data in data/processed/")
    print("  2. Run: python train.py --epochs 10")
    print("  3. Run: python benchmark.py")


if __name__ == "__main__":
    main()
