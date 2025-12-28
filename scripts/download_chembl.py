"""
ChEMBL Dataset Downloader for Drug Discovery

Downloads bioactivity data from ChEMBL database including:
- Approved drugs with SMILES
- Activity data (IC50, EC50)
- Target information
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    print("Installing chembl-webresource-client...")
    os.system("pip install chembl-webresource-client")
    from chembl_webresource_client.new_client import new_client


def download_approved_drugs(limit: int = None) -> pd.DataFrame:
    """
    Download approved drugs from ChEMBL.
    
    Args:
        limit: Maximum number of drugs to download (None for all)
    
    Returns:
        DataFrame with approved drug data
    """
    print("Downloading approved drugs from ChEMBL...")
    
    molecule = new_client.molecule
    
    # Query for approved drugs with SMILES
    approved_drugs = molecule.filter(
        max_phase=4,  # Phase 4 = Approved
        molecule_type='Small molecule'
    ).only([
        'molecule_chembl_id',
        'pref_name',
        'molecule_structures',
        'max_phase',
        'first_approval',
        'oral',
        'parenteral',
        'topical',
        'molecule_properties'
    ])
    
    drugs_list = []
    count = 0
    
    for drug in tqdm(approved_drugs, desc="Fetching approved drugs"):
        if limit and count >= limit:
            break
            
        try:
            structures = drug.get('molecule_structures', {})
            if structures and structures.get('canonical_smiles'):
                properties = drug.get('molecule_properties', {}) or {}
                
                drugs_list.append({
                    'chembl_id': drug['molecule_chembl_id'],
                    'name': drug.get('pref_name', ''),
                    'smiles': structures['canonical_smiles'],
                    'max_phase': drug.get('max_phase', 4),
                    'first_approval': drug.get('first_approval'),
                    'molecular_weight': properties.get('full_mwt'),
                    'alogp': properties.get('alogp'),
                    'hba': properties.get('hba'),
                    'hbd': properties.get('hbd'),
                    'psa': properties.get('psa'),
                    'status': 'approved'
                })
                count += 1
        except Exception as e:
            continue
    
    df = pd.DataFrame(drugs_list)
    print(f"Downloaded {len(df)} approved drugs")
    return df


def download_failed_drugs(limit: int = None) -> pd.DataFrame:
    """
    Download drugs that failed in clinical trials.
    These are drugs with max_phase < 4 that were once in development.
    
    Args:
        limit: Maximum number of drugs to download
    
    Returns:
        DataFrame with failed drug data
    """
    print("Downloading clinical trial drugs (phases 1-3)...")
    
    molecule = new_client.molecule
    
    # Get drugs in clinical phases (not approved)
    trial_drugs = molecule.filter(
        max_phase__gte=1,
        max_phase__lt=4,
        molecule_type='Small molecule'
    ).only([
        'molecule_chembl_id',
        'pref_name',
        'molecule_structures',
        'max_phase',
        'molecule_properties'
    ])
    
    drugs_list = []
    count = 0
    
    for drug in tqdm(trial_drugs, desc="Fetching trial drugs"):
        if limit and count >= limit:
            break
            
        try:
            structures = drug.get('molecule_structures', {})
            if structures and structures.get('canonical_smiles'):
                properties = drug.get('molecule_properties', {}) or {}
                
                drugs_list.append({
                    'chembl_id': drug['molecule_chembl_id'],
                    'name': drug.get('pref_name', ''),
                    'smiles': structures['canonical_smiles'],
                    'max_phase': drug.get('max_phase'),
                    'molecular_weight': properties.get('full_mwt'),
                    'alogp': properties.get('alogp'),
                    'hba': properties.get('hba'),
                    'hbd': properties.get('hbd'),
                    'psa': properties.get('psa'),
                    'status': 'failed'  # Treat non-approved as failed
                })
                count += 1
        except Exception as e:
            continue
    
    df = pd.DataFrame(drugs_list)
    print(f"Downloaded {len(df)} clinical trial drugs")
    return df


def download_bioactivity_data(target_types: list = None, limit: int = 5000) -> pd.DataFrame:
    """
    Download bioactivity data with activity measurements.
    
    Args:
        target_types: List of target types to filter
        limit: Maximum records to download
    
    Returns:
        DataFrame with bioactivity data
    """
    print("Downloading bioactivity data from ChEMBL...")
    
    activity = new_client.activity
    
    # Get activities with IC50/EC50 values
    activities = activity.filter(
        standard_type__in=['IC50', 'EC50', 'Ki', 'Kd'],
        standard_units='nM',
        standard_relation='='
    ).only([
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_type',
        'standard_value',
        'standard_units',
        'target_chembl_id',
        'target_pref_name',
        'assay_type'
    ])
    
    activity_list = []
    count = 0
    
    for act in tqdm(activities, desc="Fetching bioactivity data"):
        if count >= limit:
            break
            
        try:
            if act.get('canonical_smiles') and act.get('standard_value'):
                activity_list.append({
                    'chembl_id': act['molecule_chembl_id'],
                    'smiles': act['canonical_smiles'],
                    'activity_type': act['standard_type'],
                    'activity_value': float(act['standard_value']),
                    'activity_units': act['standard_units'],
                    'target_id': act.get('target_chembl_id', ''),
                    'target_name': act.get('target_pref_name', ''),
                    'assay_type': act.get('assay_type', '')
                })
                count += 1
        except Exception as e:
            continue
    
    df = pd.DataFrame(activity_list)
    print(f"Downloaded {len(df)} bioactivity records")
    return df


def main(test_mode: bool = False):
    """
    Main function to download all ChEMBL data.
    
    Args:
        test_mode: If True, download only small samples
    """
    output_dir = RAW_DATA_DIR / "chembl"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    limit = 100 if test_mode else None
    
    # Download approved drugs
    approved_df = download_approved_drugs(limit=limit)
    approved_df.to_csv(output_dir / "chembl_approved_drugs.csv", index=False)
    
    # Download failed/trial drugs
    failed_df = download_failed_drugs(limit=limit or 5000)
    failed_df.to_csv(output_dir / "chembl_trial_drugs.csv", index=False)
    
    # Download bioactivity data
    bioactivity_df = download_bioactivity_data(limit=500 if test_mode else 10000)
    bioactivity_df.to_csv(output_dir / "chembl_bioactivity.csv", index=False)
    
    # Summary
    print("\n" + "="*50)
    print("ChEMBL Download Complete!")
    print("="*50)
    print(f"Approved drugs: {len(approved_df)}")
    print(f"Trial drugs: {len(failed_df)}")
    print(f"Bioactivity records: {len(bioactivity_df)}")
    print(f"Output directory: {output_dir}")
    
    return {
        'approved': approved_df,
        'failed': failed_df,
        'bioactivity': bioactivity_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ChEMBL drug data")
    parser.add_argument("--test", action="store_true", help="Test mode - download small samples")
    args = parser.parse_args()
    
    main(test_mode=args.test)
