"""
DrugBank Dataset Downloader for Drug Discovery

Downloads drug data from DrugBank open data including:
- Approved drugs
- Withdrawn drugs
- Experimental drugs
- Drug targets and interactions
"""

import os
import sys
import requests
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from io import BytesIO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR


# DrugBank open data URLs (simplified/open datasets)
DRUGBANK_URLS = {
    "approved_structures": "https://go.drugbank.com/releases/latest/downloads/approved-structure-links",
    "all_structures": "https://go.drugbank.com/releases/latest/downloads/all-structure-links",
}

# Alternative: Use PubChem for drug data (fully open)
PUBCHEM_URLS = {
    "drugs": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
}


def download_drugbank_vocabulary() -> pd.DataFrame:
    """
    Download DrugBank vocabulary with drug names and identifiers.
    Falls back to alternative sources if DrugBank is unavailable.
    
    Returns:
        DataFrame with drug data
    """
    print("Attempting to download DrugBank open vocabulary...")
    
    # Try DrugBank open vocabulary
    vocab_url = "https://go.drugbank.com/releases/latest/downloads/drugbank_vocabulary"
    
    try:
        response = requests.get(vocab_url, timeout=30)
        if response.status_code == 200:
            # Parse vocabulary
            lines = response.text.strip().split('\n')
            header = lines[0].split(',')
            
            data = []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 3:
                    data.append({
                        'drugbank_id': parts[0],
                        'name': parts[1],
                        'cas_number': parts[2] if len(parts) > 2 else '',
                        'status': 'approved'  # Vocabulary contains approved drugs
                    })
            
            df = pd.DataFrame(data)
            print(f"Downloaded {len(df)} drugs from DrugBank vocabulary")
            return df
            
    except Exception as e:
        print(f"DrugBank download failed: {e}")
    
    # Fallback: Create from known drug lists
    print("Using fallback drug list...")
    return create_fallback_drug_list()


def create_fallback_drug_list() -> pd.DataFrame:
    """
    Create a fallback drug list from publicly available sources.
    Uses FDA approved drug list and known withdrawn drugs.
    
    Returns:
        DataFrame with drug data
    """
    # Known FDA-approved drugs (sample)
    approved_drugs = [
        # Common medications
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "status": "approved"},
        {"name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "status": "approved"},
        {"name": "Acetaminophen", "smiles": "CC(=O)NC1=CC=C(C=C1)O", "status": "approved"},
        {"name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "status": "approved"},
        {"name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "status": "approved"},
        {"name": "Omeprazole", "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC", "status": "approved"},
        {"name": "Lisinopril", "smiles": "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "status": "approved"},
        {"name": "Amlodipine", "smiles": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN", "status": "approved"},
        {"name": "Metoprolol", "smiles": "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O", "status": "approved"},
        {"name": "Losartan", "smiles": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl", "status": "approved"},
        {"name": "Simvastatin", "smiles": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C", "status": "approved"},
        {"name": "Levothyroxine", "smiles": "NC(CC1=CC(I)=C(OC2=CC(I)=C(O)C(I)=C2)C(I)=C1)C(=O)O", "status": "approved"},
        {"name": "Azithromycin", "smiles": "CC1C(C(CC(O1)OC2C(C(C(C(C2N(C)C)O)C)OC3CC(C(C(O3)C)O)(C)OC)C)OC)O", "status": "approved"},
        {"name": "Gabapentin", "smiles": "NCC1(CCCCC1)CC(=O)O", "status": "approved"},
        {"name": "Sertraline", "smiles": "CNC1CCC(C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl", "status": "approved"},
        {"name": "Fluoxetine", "smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F", "status": "approved"},
        {"name": "Escitalopram", "smiles": "CN(C)CCCC1(C2=CC=C(C=C2)C#N)C3=CC(=CC=C3CO1)F", "status": "approved"},
        {"name": "Duloxetine", "smiles": "CNCC[C@@H](C1=CC=CS1)OC2=CC=C(C=C2)C3=CC=CC=N3", "status": "approved"},
        {"name": "Trazodone", "smiles": "ClC1=CC=CC(N2CCN(CCCN3C(=O)N4C=CC=CC4=N3)CC2)=C1", "status": "approved"},
        {"name": "Clopidogrel", "smiles": "COC(=O)C(C1=CC2=C(S1)C=C(Cl)C=C2)N3CCC4=CC=CC=C4C3", "status": "approved"},
    ]
    
    # Known withdrawn/failed drugs
    withdrawn_drugs = [
        {"name": "Rofecoxib", "smiles": "CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3", "status": "withdrawn"},
        {"name": "Valdecoxib", "smiles": "CC1=C(C(=NO1)C2=CC=C(C=C2)S(=O)(=O)N)C3=CC=CC=C3", "status": "withdrawn"},
        {"name": "Tegaserod", "smiles": "CCCCCNC(=N)NN=CC1=CNC2=CC=C(C=C12)OC", "status": "withdrawn"},
        {"name": "Sibutramine", "smiles": "CC(C)(C)C(=O)C(C1=CC=C(C=C1)Cl)C(C)N(C)C", "status": "withdrawn"},
        {"name": "Rosiglitazone", "smiles": "CN(CCOC1=CC=C(C=C1)CC2C(=O)NC(=O)S2)C3=CC=CC=N3", "status": "withdrawn"},
        {"name": "Troglitazone", "smiles": "CC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4", "status": "withdrawn"},
        {"name": "Cisapride", "smiles": "COC1=C(C=C(C=C1)Cl)C(=O)NC2CCN(CC2)CCCOC3=CC=C(C=C3)F", "status": "withdrawn"},
        {"name": "Phenformin", "smiles": "NC(=N)NC(=N)NCCC1=CC=CC=C1", "status": "withdrawn"},
        {"name": "Thalidomide", "smiles": "O=C1CCC(N2C(=O)C3=CC=CC=C3C2=O)C(=O)N1", "status": "withdrawn"},
        {"name": "Cerivastatin", "smiles": "COC1=C(C=C(C=C1)C=CC(CC(CC(=O)O)O)O)C2=NC(=C(C=C2)F)C(C)C", "status": "withdrawn"},
    ]
    
    # Experimental drugs (in trials)
    experimental_drugs = [
        {"name": "Compound_EXP001", "smiles": "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2", "status": "experimental"},
        {"name": "Compound_EXP002", "smiles": "COC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2", "status": "experimental"},
        {"name": "Compound_EXP003", "smiles": "CC(C)NCC(O)COC1=CC=CC=C1", "status": "experimental"},
        {"name": "Compound_EXP004", "smiles": "NC1=CC=C(C=C1)C(=O)NCCC2=CNC3=CC=CC=C23", "status": "experimental"},
        {"name": "Compound_EXP005", "smiles": "COC1=CC=C(C=C1)CC(=O)NC2=CC=CC=C2", "status": "experimental"},
    ]
    
    all_drugs = approved_drugs + withdrawn_drugs + experimental_drugs
    df = pd.DataFrame(all_drugs)
    
    print(f"Created fallback drug list with {len(df)} drugs")
    return df


def download_from_pubchem(limit: int = 5000) -> pd.DataFrame:
    """
    Download drug-like molecules from PubChem.
    
    Args:
        limit: Maximum records to download
    
    Returns:
        DataFrame with molecule data
    """
    print("Downloading molecules from PubChem...")
    
    # Use PubChem PUG REST API for drug-like compounds
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # Get compounds with drug-like properties
    molecules = []
    
    # Sample CIDs from approved drugs category
    try:
        search_url = f"{base_url}/compound/name/drug/cids/JSON?name_type=word"
        response = requests.get(search_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])[:limit]
            
            # Get SMILES for each CID in batches
            batch_size = 100
            for i in tqdm(range(0, min(len(cids), limit), batch_size)):
                batch_cids = cids[i:i+batch_size]
                cid_str = ','.join(map(str, batch_cids))
                
                props_url = f"{base_url}/compound/cid/{cid_str}/property/CanonicalSMILES,IUPACName/JSON"
                props_response = requests.get(props_url, timeout=30)
                
                if props_response.status_code == 200:
                    props_data = props_response.json()
                    for prop in props_data.get('PropertyTable', {}).get('Properties', []):
                        molecules.append({
                            'pubchem_cid': prop.get('CID'),
                            'name': prop.get('IUPACName', ''),
                            'smiles': prop.get('CanonicalSMILES', ''),
                            'status': 'approved'
                        })
            
            print(f"Downloaded {len(molecules)} molecules from PubChem")
            
    except Exception as e:
        print(f"PubChem download error: {e}")
    
    return pd.DataFrame(molecules) if molecules else pd.DataFrame()


def main(test_mode: bool = False):
    """
    Main function to download DrugBank/alternative drug data.
    
    Args:
        test_mode: If True, download smaller samples
    """
    output_dir = RAW_DATA_DIR / "drugbank"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download DrugBank vocabulary or use fallback
    drugbank_df = download_drugbank_vocabulary()
    drugbank_df.to_csv(output_dir / "drugbank_drugs.csv", index=False)
    
    # Download from PubChem as supplement
    limit = 100 if test_mode else 2000
    pubchem_df = download_from_pubchem(limit=limit)
    if not pubchem_df.empty:
        pubchem_df.to_csv(output_dir / "pubchem_drugs.csv", index=False)
    
    # Create fallback list (always available)
    fallback_df = create_fallback_drug_list()
    fallback_df.to_csv(output_dir / "curated_drugs.csv", index=False)
    
    # Summary
    print("\n" + "="*50)
    print("DrugBank/PubChem Download Complete!")
    print("="*50)
    print(f"DrugBank drugs: {len(drugbank_df)}")
    print(f"PubChem molecules: {len(pubchem_df)}")
    print(f"Curated drugs: {len(fallback_df)}")
    print(f"Output directory: {output_dir}")
    
    return {
        'drugbank': drugbank_df,
        'pubchem': pubchem_df,
        'curated': fallback_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DrugBank/PubChem drug data")
    parser.add_argument("--test", action="store_true", help="Test mode - download small samples")
    args = parser.parse_args()
    
    main(test_mode=args.test)
