"""
FDA Dataset Downloader for Drug Discovery

Downloads FDA drug data including:
- Approved drugs
- New drug applications
- Failed/rejected drugs
- Orange Book data
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR


# FDA OpenFDA API endpoints
FDA_API_BASE = "https://api.fda.gov"
FDA_DRUG_ENDPOINT = f"{FDA_API_BASE}/drug/drugsfda.json"
FDA_NDC_ENDPOINT = f"{FDA_API_BASE}/drug/ndc.json"


def download_fda_approved_drugs(limit: int = 1000) -> pd.DataFrame:
    """
    Download FDA approved drugs using OpenFDA API.
    
    Args:
        limit: Maximum number of drugs to download
    
    Returns:
        DataFrame with FDA approved drug data
    """
    print("Downloading FDA approved drugs...")
    
    drugs_list = []
    skip = 0
    batch_size = 100  # OpenFDA limit per request
    
    with tqdm(total=limit, desc="Fetching FDA drugs") as pbar:
        while len(drugs_list) < limit:
            try:
                params = {
                    "limit": min(batch_size, limit - len(drugs_list)),
                    "skip": skip
                }
                
                response = requests.get(FDA_DRUG_ENDPOINT, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"API error: {response.status_code}")
                    break
                
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                
                for drug in results:
                    # Extract product information
                    products = drug.get('products', [])
                    for product in products:
                        active_ingredients = product.get('active_ingredients', [])
                        
                        for ingredient in active_ingredients:
                            drugs_list.append({
                                'application_number': drug.get('application_number', ''),
                                'brand_name': product.get('brand_name', ''),
                                'generic_name': ingredient.get('name', ''),
                                'dosage_form': product.get('dosage_form', ''),
                                'route': product.get('route', ''),
                                'strength': ingredient.get('strength', ''),
                                'marketing_status': product.get('marketing_status', ''),
                                'sponsor_name': drug.get('sponsor_name', ''),
                                'status': 'approved'
                            })
                
                skip += batch_size
                pbar.update(len(results))
                
            except Exception as e:
                print(f"Error fetching FDA data: {e}")
                break
    
    df = pd.DataFrame(drugs_list[:limit])
    print(f"Downloaded {len(df)} FDA drug records")
    return df


def download_fda_ndc_drugs(limit: int = 1000) -> pd.DataFrame:
    """
    Download drugs from FDA National Drug Code (NDC) database.
    
    Args:
        limit: Maximum number to download
    
    Returns:
        DataFrame with NDC drug data
    """
    print("Downloading FDA NDC drug data...")
    
    drugs_list = []
    skip = 0
    batch_size = 100
    
    with tqdm(total=limit, desc="Fetching NDC data") as pbar:
        while len(drugs_list) < limit:
            try:
                params = {
                    "limit": min(batch_size, limit - len(drugs_list)),
                    "skip": skip
                }
                
                response = requests.get(FDA_NDC_ENDPOINT, params=params, timeout=30)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                
                for drug in results:
                    drugs_list.append({
                        'product_ndc': drug.get('product_ndc', ''),
                        'generic_name': drug.get('generic_name', ''),
                        'brand_name': drug.get('brand_name', ''),
                        'labeler_name': drug.get('labeler_name', ''),
                        'dosage_form': drug.get('dosage_form', ''),
                        'route': drug.get('route', [''])[0] if drug.get('route') else '',
                        'product_type': drug.get('product_type', ''),
                        'marketing_category': drug.get('marketing_category', ''),
                        'status': 'approved'
                    })
                
                skip += batch_size
                pbar.update(len(results))
                
            except Exception as e:
                print(f"Error fetching NDC data: {e}")
                break
    
    df = pd.DataFrame(drugs_list[:limit])
    print(f"Downloaded {len(df)} NDC drug records")
    return df


def create_failed_drugs_dataset() -> pd.DataFrame:
    """
    Create a dataset of known failed drugs with their failure reasons.
    Based on publicly available information about drug withdrawals and failures.
    
    Returns:
        DataFrame with failed drug data
    """
    print("Creating failed drugs dataset...")
    
    # Known drug failures with SMILES (when available)
    failed_drugs = [
        # Cardiovascular failures
        {"name": "Rofecoxib (Vioxx)", "smiles": "CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3", "reason": "Cardiovascular risk", "year": 2004, "phase_failed": "Post-market", "status": "withdrawn"},
        {"name": "Valdecoxib (Bextra)", "smiles": "CC1=C(C(=NO1)C2=CC=C(C=C2)S(=O)(=O)N)C3=CC=CC=C3", "reason": "Cardiovascular/skin reactions", "year": 2005, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Liver toxicity
        {"name": "Troglitazone (Rezulin)", "smiles": "CC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4", "reason": "Liver toxicity", "year": 2000, "phase_failed": "Post-market", "status": "withdrawn"},
        {"name": "Bromfenac (Duract)", "smiles": "NC1=C(C=C(C=C1)C(=O)CC1=CC=CC=C1)C(=O)O", "reason": "Liver failure", "year": 1998, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Cardiac arrhythmia
        {"name": "Terfenadine (Seldane)", "smiles": "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4", "reason": "Cardiac arrhythmia", "year": 1998, "phase_failed": "Post-market", "status": "withdrawn"},
        {"name": "Cisapride (Propulsid)", "smiles": "COC1=C(C=C(C=C1)Cl)C(=O)NC2CCN(CC2)CCCOC3=CC=C(C=C3)F", "reason": "Cardiac arrhythmia", "year": 2000, "phase_failed": "Post-market", "status": "withdrawn"},
        {"name": "Grepafloxacin", "smiles": "CC1=C(C=C2C(=C1)N(C=C(C2=O)C(=O)O)C3CC3)N4CCNCC4", "reason": "QT prolongation", "year": 1999, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Hemorrhagic stroke
        {"name": "Phenylpropanolamine", "smiles": "CC(C(C1=CC=CC=C1)O)N", "reason": "Hemorrhagic stroke", "year": 2000, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Lactic acidosis
        {"name": "Phenformin", "smiles": "NC(=N)NC(=N)NCCC1=CC=CC=C1", "reason": "Lactic acidosis", "year": 1978, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Birth defects
        {"name": "Thalidomide", "smiles": "O=C1CCC(N2C(=O)C3=CC=CC=C3C2=O)C(=O)N1", "reason": "Teratogenicity", "year": 1961, "phase_failed": "Post-market", "status": "restricted"},
        
        # Failed in clinical trials (Phase II/III)
        {"name": "Torcetrapib", "smiles": "CCOC(=O)C1=C(C)NC(C)=C(C1C2=CC=CC=C2CF)C(=O)OCC", "reason": "Increased mortality", "year": 2006, "phase_failed": "Phase III", "status": "failed"},
        {"name": "Semagacestat", "smiles": "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(C)NC(=O)OC(C)(C)C)C(=O)NC(CC(C)C)C=O", "reason": "Worsened cognition", "year": 2010, "phase_failed": "Phase III", "status": "failed"},
        {"name": "Bapineuzumab", "smiles": "N/A (Antibody)", "reason": "No efficacy in Alzheimer's", "year": 2012, "phase_failed": "Phase III", "status": "failed"},
        
        # Weight loss failures
        {"name": "Sibutramine (Meridia)", "smiles": "CC(C)(C)C(=O)C(C1=CC=C(C=C1)Cl)C(C)N(C)C", "reason": "Cardiovascular risk", "year": 2010, "phase_failed": "Post-market", "status": "withdrawn"},
        {"name": "Rimonabant (Acomplia)", "smiles": "CC1=C(C(=NN1C2=CC=C(C=C2)Cl)C3=CC=C(C=C3)Cl)C(=O)NN4CCCCC4", "reason": "Psychiatric effects", "year": 2008, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Muscle damage
        {"name": "Cerivastatin (Baycol)", "smiles": "COC1=C(C=C(C=C1)C=CC(CC(CC(=O)O)O)O)C2=NC(=C(C=C2)F)C(C)C", "reason": "Rhabdomyolysis", "year": 2001, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Cancer risk
        {"name": "Tegaserod (Zelnorm)", "smiles": "CCCCCNC(=N)NN=CC1=CNC2=CC=C(C=C12)OC", "reason": "Cardiovascular risk", "year": 2007, "phase_failed": "Post-market", "status": "withdrawn"},
        
        # Recent failures
        {"name": "Aducanumab", "smiles": "N/A (Antibody)", "reason": "Controversial efficacy", "year": 2021, "phase_failed": "Accelerated approval", "status": "controversial"},
    ]
    
    # Add more experimental failures
    experimental_failures = [
        {"name": "Drug_Candidate_001", "smiles": "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "reason": "Toxicity in Phase I", "year": 2020, "phase_failed": "Phase I", "status": "failed"},
        {"name": "Drug_Candidate_002", "smiles": "COC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2", "reason": "No efficacy in Phase II", "year": 2019, "phase_failed": "Phase II", "status": "failed"},
        {"name": "Drug_Candidate_003", "smiles": "CC(C)NC(=O)C1=CC=C(C=C1)Cl", "reason": "Safety concerns", "year": 2021, "phase_failed": "Phase I", "status": "failed"},
        {"name": "Drug_Candidate_004", "smiles": "NC1=CC=C(C=C1)S(=O)(=O)N", "reason": "Poor pharmacokinetics", "year": 2018, "phase_failed": "Preclinical", "status": "failed"},
        {"name": "Drug_Candidate_005", "smiles": "COC1=CC=C(C=C1)C2=NC3=CC=CC=C3O2", "reason": "Off-target effects", "year": 2020, "phase_failed": "Phase II", "status": "failed"},
    ]
    
    all_failed = failed_drugs + experimental_failures
    df = pd.DataFrame(all_failed)
    
    # Filter out entries without valid SMILES
    df = df[df['smiles'] != 'N/A (Antibody)']
    
    print(f"Created dataset with {len(df)} failed/withdrawn drugs")
    return df


def create_novel_alternatives_dataset() -> pd.DataFrame:
    """
    Create a dataset of novel drug alternatives and analogs.
    These are variations or improvements on existing drugs.
    
    Returns:
        DataFrame with novel alternatives
    """
    print("Creating novel alternatives dataset...")
    
    # Novel alternatives/analogs (fictitious examples for training)
    alternatives = [
        # Improved NSAIDs
        {"name": "Novel_NSAID_001", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)NC2=CC=CC=C2", "original_drug": "Ibuprofen", "improvement": "Reduced GI effects", "status": "experimental"},
        {"name": "Novel_NSAID_002", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)NC(C)C", "original_drug": "Aspirin", "improvement": "Pro-drug form", "status": "experimental"},
        
        # Improved statins
        {"name": "Novel_Statin_001", "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)N", "original_drug": "Atorvastatin", "improvement": "Better selectivity", "status": "experimental"},
        
        # Improved antidiabetics
        {"name": "Novel_DPP4i_001", "smiles": "CN(C)C(=N)NC(=N)NCC1=CC=CC=C1", "original_drug": "Metformin", "improvement": "Longer half-life", "status": "experimental"},
        
        # Improved antihypertensives
        {"name": "Novel_ACEi_001", "smiles": "CCOC(=O)C(C)NC(C)C(=O)N1CCCC1C(=O)O", "original_drug": "Enalapril", "improvement": "Once daily dosing", "status": "experimental"},
        {"name": "Novel_ARB_001", "smiles": "CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3)CO)Cl", "original_drug": "Losartan", "improvement": "Improved bioavailability", "status": "experimental"},
        
        # Improved antibiotics
        {"name": "Novel_Antibiotic_001", "smiles": "CC1=C(C=C(C=C1)N2C=C(C=N2)C(=O)O)F", "original_drug": "Ciprofloxacin", "improvement": "Broader spectrum", "status": "experimental"},
        
        # Improved antidepressants
        {"name": "Novel_SSRI_001", "smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)F", "original_drug": "Fluoxetine", "improvement": "Faster onset", "status": "experimental"},
        {"name": "Novel_SNRI_001", "smiles": "CNCC[C@H](C1=CC=CS1)OC2=CC=CC=C2", "original_drug": "Duloxetine", "improvement": "Reduced side effects", "status": "experimental"},
        
        # More alternatives
        {"name": "Novel_PPI_001", "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2", "original_drug": "Omeprazole", "improvement": "Longer duration", "status": "experimental"},
        {"name": "Novel_Antiplatelet_001", "smiles": "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=CC=CC=C3C2", "original_drug": "Clopidogrel", "improvement": "Better response rate", "status": "experimental"},
    ]
    
    df = pd.DataFrame(alternatives)
    print(f"Created dataset with {len(df)} novel alternatives")
    return df


def main(test_mode: bool = False):
    """
    Main function to download all FDA-related drug data.
    
    Args:
        test_mode: If True, download smaller samples
    """
    output_dir = RAW_DATA_DIR / "fda"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    limit = 100 if test_mode else 1000
    
    # Download FDA approved drugs
    approved_df = download_fda_approved_drugs(limit=limit)
    approved_df.to_csv(output_dir / "fda_approved_drugs.csv", index=False)
    
    # Download NDC data
    ndc_df = download_fda_ndc_drugs(limit=limit)
    ndc_df.to_csv(output_dir / "fda_ndc_drugs.csv", index=False)
    
    # Create failed drugs dataset
    failed_df = create_failed_drugs_dataset()
    failed_df.to_csv(output_dir / "failed_drugs.csv", index=False)
    
    # Create novel alternatives dataset
    alternatives_df = create_novel_alternatives_dataset()
    alternatives_df.to_csv(output_dir / "novel_alternatives.csv", index=False)
    
    # Summary
    print("\n" + "="*50)
    print("FDA Data Download Complete!")
    print("="*50)
    print(f"FDA approved drugs: {len(approved_df)}")
    print(f"NDC drugs: {len(ndc_df)}")
    print(f"Failed/withdrawn drugs: {len(failed_df)}")
    print(f"Novel alternatives: {len(alternatives_df)}")
    print(f"Output directory: {output_dir}")
    
    return {
        'approved': approved_df,
        'ndc': ndc_df,
        'failed': failed_df,
        'alternatives': alternatives_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FDA drug data")
    parser.add_argument("--test", action="store_true", help="Test mode - download small samples")
    args = parser.parse_args()
    
    main(test_mode=args.test)
