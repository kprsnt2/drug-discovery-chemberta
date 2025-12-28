"""
Dataset Preparation Script for Drug Discovery

Combines all downloaded datasets into unified training data:
- Merges ChEMBL, DrugBank, FDA data
- Validates and cleans SMILES
- Creates train/val/test splits
- Generates classification labels
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_CONFIG

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Using basic SMILES validation.")
    RDKIT_AVAILABLE = False


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    if len(smiles) < DATASET_CONFIG['min_smiles_length']:
        return False
    
    if len(smiles) > DATASET_CONFIG['max_smiles_length']:
        return False
    
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    # Basic validation without RDKit
    valid_chars = set('CNOSPFClBrI[]()=#+-0123456789@/\\cnops')
    return all(c in valid_chars for c in smiles)


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form for deduplication.
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Canonical SMILES or original if conversion fails
    """
    if not RDKIT_AVAILABLE:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    
    return smiles


def augment_smiles(smiles: str, num_augments: int = 2) -> list:
    """
    Generate SMILES augmentations (different valid representations).
    
    Args:
        smiles: Input SMILES
        num_augments: Number of augmentations to generate
    
    Returns:
        List of augmented SMILES
    """
    if not RDKIT_AVAILABLE:
        return [smiles]
    
    augmented = [smiles]
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for i in range(num_augments):
                # Random SMILES enumeration
                random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                if random_smiles and random_smiles not in augmented:
                    augmented.append(random_smiles)
    except:
        pass
    
    return augmented


def load_chembl_data() -> pd.DataFrame:
    """Load and process ChEMBL data."""
    chembl_dir = RAW_DATA_DIR / "chembl"
    dfs = []
    
    # Load approved drugs
    approved_path = chembl_dir / "chembl_approved_drugs.csv"
    if approved_path.exists():
        df = pd.read_csv(approved_path)
        df['source'] = 'chembl'
        df['label'] = 1  # Approved
        dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    # Load trial drugs (failed)
    trial_path = chembl_dir / "chembl_trial_drugs.csv"
    if trial_path.exists():
        df = pd.read_csv(trial_path)
        df['source'] = 'chembl'
        df['label'] = 0  # Failed/Not approved
        dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_drugbank_data() -> pd.DataFrame:
    """Load and process DrugBank data."""
    drugbank_dir = RAW_DATA_DIR / "drugbank"
    dfs = []
    
    # Load curated drugs
    curated_path = drugbank_dir / "curated_drugs.csv"
    if curated_path.exists():
        df = pd.read_csv(curated_path)
        df['source'] = 'drugbank'
        df['label'] = df['status'].map(DATASET_CONFIG['label_mapping']).fillna(0).astype(int)
        dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    # Load PubChem drugs
    pubchem_path = drugbank_dir / "pubchem_drugs.csv"
    if pubchem_path.exists():
        df = pd.read_csv(pubchem_path)
        df['source'] = 'pubchem'
        df['label'] = 1  # Assume approved
        dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_fda_data() -> pd.DataFrame:
    """Load and process FDA data."""
    fda_dir = RAW_DATA_DIR / "fda"
    dfs = []
    
    # Load failed drugs
    failed_path = fda_dir / "failed_drugs.csv"
    if failed_path.exists():
        df = pd.read_csv(failed_path)
        df['source'] = 'fda_failed'
        df['label'] = 0  # Failed
        if 'smiles' in df.columns and 'name' in df.columns:
            dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    # Load novel alternatives
    alternatives_path = fda_dir / "novel_alternatives.csv"
    if alternatives_path.exists():
        df = pd.read_csv(alternatives_path)
        df['source'] = 'fda_novel'
        df['label'] = 0  # Experimental (not yet approved)
        if 'smiles' in df.columns and 'name' in df.columns:
            dfs.append(df[['smiles', 'name', 'status', 'source', 'label']])
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def prepare_dataset(augment: bool = True, test_mode: bool = False):
    """
    Prepare the complete dataset for training.
    
    Args:
        augment: Whether to apply SMILES augmentation
        test_mode: If True, use smaller subset
    """
    print("="*50)
    print("Preparing Drug Discovery Dataset")
    print("="*50)
    
    # Load all data sources
    print("\n[1/6] Loading data sources...")
    chembl_df = load_chembl_data()
    drugbank_df = load_drugbank_data()
    fda_df = load_fda_data()
    
    print(f"  ChEMBL: {len(chembl_df)} records")
    print(f"  DrugBank: {len(drugbank_df)} records")
    print(f"  FDA: {len(fda_df)} records")
    
    # Combine all data
    print("\n[2/6] Combining datasets...")
    all_data = pd.concat([chembl_df, drugbank_df, fda_df], ignore_index=True)
    print(f"  Total records: {len(all_data)}")
    
    # Validate SMILES
    print("\n[3/6] Validating SMILES...")
    valid_mask = all_data['smiles'].apply(validate_smiles)
    all_data = all_data[valid_mask].copy()
    print(f"  Valid records: {len(all_data)}")
    
    # Canonicalize and deduplicate
    print("\n[4/6] Canonicalizing and deduplicating...")
    all_data['smiles'] = all_data['smiles'].apply(canonicalize_smiles)
    all_data = all_data.drop_duplicates(subset=['smiles'], keep='first')
    print(f"  Unique molecules: {len(all_data)}")
    
    # Balance classes if needed
    print("\n[5/6] Balancing classes...")
    approved = all_data[all_data['label'] == 1]
    failed = all_data[all_data['label'] == 0]
    
    print(f"  Approved drugs: {len(approved)}")
    print(f"  Failed drugs: {len(failed)}")
    
    # Augment minority class if imbalanced
    if augment and DATASET_CONFIG['smiles_augmentation']:
        min_class = approved if len(approved) < len(failed) else failed
        max_class = failed if len(approved) < len(failed) else approved
        
        if len(min_class) > 0 and len(max_class) / len(min_class) > 2:
            print("  Augmenting minority class...")
            augmented_rows = []
            for _, row in tqdm(min_class.iterrows(), total=len(min_class)):
                aug_smiles = augment_smiles(row['smiles'], 
                                           DATASET_CONFIG['augmentation_factor'])
                for smi in aug_smiles[1:]:  # Skip original
                    new_row = row.copy()
                    new_row['smiles'] = smi
                    new_row['source'] = row['source'] + '_aug'
                    augmented_rows.append(new_row)
            
            if augmented_rows:
                aug_df = pd.DataFrame(augmented_rows)
                all_data = pd.concat([all_data, aug_df], ignore_index=True)
    
    # For test mode, use smaller subset
    if test_mode:
        all_data = all_data.sample(n=min(500, len(all_data)), random_state=42)
    
    # Create train/val/test splits
    print("\n[6/6] Creating data splits...")
    train_ratio = DATASET_CONFIG['train_ratio']
    val_ratio = DATASET_CONFIG['val_ratio']
    
    # Stratified split
    train_df, temp_df = train_test_split(
        all_data, 
        test_size=(1 - train_ratio),
        stratify=all_data['label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )
    
    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    all_data.to_csv(PROCESSED_DATA_DIR / "full_dataset.csv", index=False)
    
    # Save dataset statistics
    stats = {
        'total_samples': len(all_data),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'approved_samples': int((all_data['label'] == 1).sum()),
        'failed_samples': int((all_data['label'] == 0).sum()),
        'sources': all_data['source'].value_counts().to_dict()
    }
    
    stats_path = PROCESSED_DATA_DIR / "dataset_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Preparation Complete!")
    print("="*50)
    print(f"Total samples: {len(all_data)}")
    print(f"  - Train: {len(train_df)} ({100*train_ratio:.0f}%)")
    print(f"  - Validation: {len(val_df)} ({100*val_ratio:.0f}%)")
    print(f"  - Test: {len(test_df)} ({100*DATASET_CONFIG['test_ratio']:.0f}%)")
    print(f"\nClass distribution:")
    print(f"  - Approved (label=1): {stats['approved_samples']}")
    print(f"  - Failed (label=0): {stats['failed_samples']}")
    print(f"\nOutput directory: {PROCESSED_DATA_DIR}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare drug discovery dataset")
    parser.add_argument("--no-augment", action="store_true", help="Disable SMILES augmentation")
    parser.add_argument("--test", action="store_true", help="Test mode - use smaller subset")
    args = parser.parse_args()
    
    prepare_dataset(augment=not args.no_augment, test_mode=args.test)
