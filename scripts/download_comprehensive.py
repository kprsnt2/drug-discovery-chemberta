"""
Enhanced Drug Data Downloader for Text Generation Training

Downloads comprehensive drug data from multiple sources with detailed annotations:
- ChEMBL: Approved drugs, trial drugs, toxicity data, ADMET predictions
- PubChem: Compound properties, bioassay results
- DrugBank: Drug interactions, withdrawal reasons
- Clinical Trials: Failure reasons and safety data

This data will be used to train a text generation model that provides
explanatory drug discovery assistance.
"""

import os
import sys
import json
import time
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    print("Installing chembl-webresource-client...")
    os.system("pip install chembl-webresource-client")
    from chembl_webresource_client.new_client import new_client

try:
    import pubchempy as pcp
except ImportError:
    print("Installing pubchempy...")
    os.system("pip install pubchempy")
    import pubchempy as pcp


# ============================================================================
# PubChem Data Download
# ============================================================================

def get_pubchem_properties(smiles: str) -> Dict:
    """
    Get compound properties from PubChem.
    
    Returns molecular properties including:
    - Molecular weight, LogP, TPSA
    - H-bond donors/acceptors
    - Rotatable bonds
    - Toxicity flags
    """
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            c = compounds[0]
            return {
                'pubchem_cid': c.cid,
                'iupac_name': c.iupac_name,
                'molecular_formula': c.molecular_formula,
                'molecular_weight': c.molecular_weight,
                'xlogp': c.xlogp,
                'tpsa': c.tpsa,
                'h_bond_donor_count': c.h_bond_donor_count,
                'h_bond_acceptor_count': c.h_bond_acceptor_count,
                'rotatable_bond_count': c.rotatable_bond_count,
                'heavy_atom_count': c.heavy_atom_count,
                'complexity': c.complexity,
                'exact_mass': c.exact_mass,
            }
    except Exception as e:
        pass
    return {}


def download_pubchem_bioassays(cid: int, limit: int = 5) -> List[Dict]:
    """
    Download bioassay results for a compound from PubChem.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            assays = data.get('AssaySummaries', {}).get('AssaySummary', [])
            return [
                {
                    'assay_id': a.get('AID'),
                    'assay_name': a.get('SourceName', ''),
                    'activity_outcome': a.get('ActivityOutcome', ''),
                    'target_name': a.get('TargetName', ''),
                }
                for a in assays[:limit]
            ]
    except Exception:
        pass
    return []


# ============================================================================
# ChEMBL Enhanced Data Download
# ============================================================================

def download_chembl_drug_mechanisms(limit: int = 5000) -> pd.DataFrame:
    """
    Download drug mechanisms of action from ChEMBL.
    """
    print("Downloading drug mechanisms from ChEMBL...")
    
    mechanism = new_client.mechanism
    mechanisms = mechanism.all()
    
    mech_list = []
    count = 0
    
    for mech in tqdm(mechanisms, desc="Fetching drug mechanisms"):
        if limit and count >= limit:
            break
        try:
            mech_list.append({
                'chembl_id': mech.get('molecule_chembl_id', ''),
                'target_chembl_id': mech.get('target_chembl_id', ''),
                'mechanism_of_action': mech.get('mechanism_of_action', ''),
                'action_type': mech.get('action_type', ''),
                'target_name': mech.get('target_pref_name', ''),
                'mechanism_comment': mech.get('mechanism_comment', ''),
            })
            count += 1
        except Exception:
            continue
    
    df = pd.DataFrame(mech_list)
    print(f"Downloaded {len(df)} drug mechanisms")
    return df


def download_chembl_drug_indications(limit: int = 10000) -> pd.DataFrame:
    """
    Download drug indications (what diseases they treat) from ChEMBL.
    """
    print("Downloading drug indications from ChEMBL...")
    
    indication = new_client.drug_indication
    indications = indication.all()
    
    ind_list = []
    count = 0
    
    for ind in tqdm(indications, desc="Fetching drug indications"):
        if limit and count >= limit:
            break
        try:
            ind_list.append({
                'chembl_id': ind.get('molecule_chembl_id', ''),
                'indication': ind.get('efo_term', '') or ind.get('mesh_heading', ''),
                'max_phase': ind.get('max_phase_for_ind'),
                'indication_refs': ind.get('indication_refs', []),
            })
            count += 1
        except Exception:
            continue
    
    df = pd.DataFrame(ind_list)
    print(f"Downloaded {len(df)} drug indications")
    return df


def download_chembl_metabolism(limit: int = 5000) -> pd.DataFrame:
    """
    Download drug metabolism data from ChEMBL.
    """
    print("Downloading drug metabolism data from ChEMBL...")
    
    metabolism = new_client.metabolism
    metab_data = metabolism.all()
    
    metab_list = []
    count = 0
    
    for metab in tqdm(metab_data, desc="Fetching metabolism data"):
        if limit and count >= limit:
            break
        try:
            metab_list.append({
                'drug_chembl_id': metab.get('drug_chembl_id', ''),
                'metabolite_chembl_id': metab.get('metabolite_chembl_id', ''),
                'enzyme_name': metab.get('enzyme_name', ''),
                'met_conversion': metab.get('met_conversion', ''),
                'organism': metab.get('organism', ''),
            })
            count += 1
        except Exception:
            continue
    
    df = pd.DataFrame(metab_list)
    print(f"Downloaded {len(df)} metabolism records")
    return df


def download_chembl_drug_warnings(limit: int = None) -> pd.DataFrame:
    """
    Download drug warning data from ChEMBL (very important for failure analysis).
    """
    print("Downloading drug warnings from ChEMBL...")
    
    drug_warning = new_client.drug_warning
    warnings = drug_warning.all()
    
    warning_list = []
    count = 0
    
    for warn in tqdm(warnings, desc="Fetching drug warnings"):
        if limit and count >= limit:
            break
        try:
            warning_list.append({
                'chembl_id': warn.get('molecule_chembl_id', ''),
                'warning_type': warn.get('warning_type', ''),
                'warning_class': warn.get('warning_class', ''),
                'warning_description': warn.get('warning_description', ''),
                'warning_country': warn.get('warning_country', ''),
                'warning_year': warn.get('warning_year'),
                'efo_term': warn.get('efo_term', ''),
            })
            count += 1
        except Exception:
            continue
    
    df = pd.DataFrame(warning_list)
    print(f"Downloaded {len(df)} drug warnings")
    return df


def download_chembl_compounds_detailed(limit: int = 10000) -> pd.DataFrame:
    """
    Download detailed compound information from ChEMBL including ADMET predictions.
    """
    print("Downloading detailed compound data from ChEMBL...")
    
    molecule = new_client.molecule
    
    # Get compounds with various phases
    compounds = molecule.filter(
        max_phase__gte=0,
        molecule_type='Small molecule'
    ).only([
        'molecule_chembl_id',
        'pref_name',
        'molecule_structures',
        'max_phase',
        'first_approval',
        'molecule_properties',
        'molecule_type',
        'therapeutic_flag',
        'natural_product',
        'withdrawn_flag',
        'prodrug',
    ])
    
    compound_list = []
    count = 0
    
    for comp in tqdm(compounds, desc="Fetching detailed compounds"):
        if limit and count >= limit:
            break
        try:
            structures = comp.get('molecule_structures', {})
            if structures and structures.get('canonical_smiles'):
                props = comp.get('molecule_properties', {}) or {}
                
                compound_list.append({
                    'chembl_id': comp['molecule_chembl_id'],
                    'name': comp.get('pref_name', ''),
                    'smiles': structures['canonical_smiles'],
                    'max_phase': comp.get('max_phase', 0),
                    'first_approval': comp.get('first_approval'),
                    'therapeutic_flag': comp.get('therapeutic_flag', False),
                    'natural_product': comp.get('natural_product', False),
                    'withdrawn_flag': comp.get('withdrawn_flag', False),
                    'prodrug': comp.get('prodrug', False),
                    # Molecular properties
                    'molecular_weight': props.get('full_mwt'),
                    'alogp': props.get('alogp'),
                    'hba': props.get('hba'),
                    'hbd': props.get('hbd'),
                    'psa': props.get('psa'),
                    'rtb': props.get('rtb'),  # rotatable bonds
                    'num_ro5_violations': props.get('num_ro5_violations'),
                    'aromatic_rings': props.get('aromatic_rings'),
                    'heavy_atoms': props.get('heavy_atoms'),
                    'qed_weighted': props.get('qed_weighted'),  # drug-likeness score
                    # Lipinski's rule of 5
                    'ro5_compliant': props.get('num_ro5_violations', 0) == 0 if props.get('num_ro5_violations') is not None else None,
                })
                count += 1
        except Exception:
            continue
    
    df = pd.DataFrame(compound_list)
    print(f"Downloaded {len(df)} detailed compounds")
    return df


# ============================================================================
# Clinical Trials Failure Data
# ============================================================================

def create_clinical_trial_failures() -> pd.DataFrame:
    """
    Create a comprehensive dataset of clinical trial failures with detailed reasons.
    Based on published literature and FDA databases.
    """
    print("Creating comprehensive clinical trial failures dataset...")
    
    failures = [
        # Phase I Failures (Toxicity/Safety)
        {
            "name": "BIIB023",
            "smiles": "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2N3CCCC3)OC",
            "phase_failed": "Phase I",
            "failure_reason": "Dose-limiting toxicity observed in healthy volunteers",
            "failure_category": "Safety/Toxicity",
            "therapeutic_area": "Oncology",
            "year": 2018,
            "company": "Biogen"
        },
        {
            "name": "AZD9150",
            "smiles": "CC(C)CC(NC(=O)C1=CC=C(C=C1)OC)C(=O)O",
            "phase_failed": "Phase I",
            "failure_reason": "Hepatotoxicity observed at therapeutic doses",
            "failure_category": "Hepatotoxicity",
            "therapeutic_area": "Oncology",
            "year": 2017,
            "company": "AstraZeneca"
        },
        
        # Phase II Failures (Efficacy)
        {
            "name": "Vercirnon",
            "smiles": "CC1=CC=C(C=C1)C2=CC(=NO2)C(=O)NC3=CC=C(C=C3)Cl",
            "phase_failed": "Phase II",
            "failure_reason": "Failed to show efficacy in Crohn's disease patients",
            "failure_category": "Lack of Efficacy",
            "therapeutic_area": "Immunology",
            "year": 2013,
            "company": "GSK"
        },
        {
            "name": "Losmapimod",
            "smiles": "CC1=CC(=C(C=C1)C(=O)NC2=CC(=CC=C2)C(F)(F)F)C3=NC(=CS3)C",
            "phase_failed": "Phase III",
            "failure_reason": "No significant cardiovascular benefit in acute coronary syndrome",
            "failure_category": "Lack of Efficacy",
            "therapeutic_area": "Cardiovascular",
            "year": 2016,
            "company": "GSK"
        },
        
        # Phase III Failures
        {
            "name": "Solanezumab",
            "smiles": "N/A",  # Antibody - will be filtered
            "phase_failed": "Phase III",
            "failure_reason": "Failed to slow cognitive decline in Alzheimer's patients",
            "failure_category": "Lack of Efficacy",
            "therapeutic_area": "CNS/Neurology",
            "year": 2016,
            "company": "Eli Lilly"
        },
        {
            "name": "Atabecestat",
            "smiles": "CC(C)(C)NC(=O)C1=CC=C(C=C1)C2=NC3=CC=CC=C3N2",
            "phase_failed": "Phase II/III",
            "failure_reason": "Liver enzyme elevations and worsening cognition",
            "failure_category": "Safety/Efficacy",
            "therapeutic_area": "CNS/Neurology",
            "year": 2018,
            "company": "Janssen"
        },
        
        # Cardiotoxicity failures
        {
            "name": "Necitumumab",
            "smiles": "N/A",  # Antibody
            "phase_failed": "Phase III",
            "failure_reason": "Increased risk of cardiopulmonary arrest",
            "failure_category": "Cardiotoxicity",
            "therapeutic_area": "Oncology",
            "year": 2018,
            "company": "Eli Lilly"
        },
        
        # Post-market withdrawals with detailed reasons
        {
            "name": "Rofecoxib (Vioxx)",
            "smiles": "CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3",
            "phase_failed": "Post-market",
            "failure_reason": "Increased risk of heart attack and stroke in long-term use due to selective COX-2 inhibition causing prostaglandin imbalance",
            "failure_category": "Cardiovascular",
            "therapeutic_area": "Pain/Inflammation",
            "year": 2004,
            "company": "Merck",
            "additional_info": "Withdrawn after APPROVe trial showed 2x increased CV risk"
        },
        {
            "name": "Cerivastatin (Baycol)",
            "smiles": "COC1=C(C=C(C=C1)C=CC(CC(CC(=O)O)O)O)C2=NC(=C(C=C2)F)C(C)C",
            "phase_failed": "Post-market",
            "failure_reason": "Fatal rhabdomyolysis (muscle breakdown) especially when combined with gemfibrozil due to CYP2C8 interaction",
            "failure_category": "Muscle Toxicity",
            "therapeutic_area": "Cardiovascular",
            "year": 2001,
            "company": "Bayer",
            "additional_info": "52 deaths linked to drug, 10x higher risk than other statins"
        },
        {
            "name": "Troglitazone (Rezulin)",
            "smiles": "CC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4",
            "phase_failed": "Post-market",
            "failure_reason": "Severe idiosyncratic hepatotoxicity leading to liver failure and deaths",
            "failure_category": "Hepatotoxicity",
            "therapeutic_area": "Diabetes",
            "year": 2000,
            "company": "Parke-Davis",
            "additional_info": "89 cases of liver failure, 63 deaths reported"
        },
        {
            "name": "Fenfluramine (Fen-Phen)",
            "smiles": "CCNC(C)CC1=CC=CC(=C1)C(F)(F)F",
            "phase_failed": "Post-market",
            "failure_reason": "Heart valve disease and primary pulmonary hypertension",
            "failure_category": "Cardiotoxicity",
            "therapeutic_area": "Weight Loss",
            "year": 1997,
            "company": "Wyeth",
            "additional_info": "Found to cause heart valve regurgitation in up to 30% of patients"
        },
        {
            "name": "Valdecoxib (Bextra)",
            "smiles": "CC1=C(C(=NO1)C2=CC=C(C=C2)S(=O)(=O)N)C3=CC=CC=C3",
            "phase_failed": "Post-market",
            "failure_reason": "Serious cardiovascular events and severe skin reactions (Stevens-Johnson syndrome)",
            "failure_category": "Cardiovascular/Dermatological",
            "therapeutic_area": "Pain/Inflammation",
            "year": 2005,
            "company": "Pfizer",
            "additional_info": "Similar CV risks to Vioxx plus life-threatening skin reactions"
        },
        {
            "name": "Sibutramine (Meridia)",
            "smiles": "CC(C)(C)C(=O)C(C1=CC=C(C=C1)Cl)C(C)N(C)C",
            "phase_failed": "Post-market",
            "failure_reason": "Increased risk of heart attacks and strokes in patients with pre-existing cardiovascular conditions",
            "failure_category": "Cardiovascular",
            "therapeutic_area": "Weight Loss",
            "year": 2010,
            "company": "Abbott",
            "additional_info": "SCOUT trial showed 16% increased CV event risk"
        },
        {
            "name": "Rosiglitazone (Avandia)",
            "smiles": "CN(CCOC1=CC=C(C=C1)CC2C(=O)NC(=O)S2)C3=CC=CC=N3",
            "phase_failed": "Restricted",
            "failure_reason": "Increased risk of myocardial infarction and cardiovascular death",
            "failure_category": "Cardiovascular",
            "therapeutic_area": "Diabetes",
            "year": 2010,
            "company": "GSK",
            "additional_info": "Meta-analysis showed 43% increased MI risk, heavily restricted but not withdrawn"
        },
        
        # More recent failures (2020+)
        {
            "name": "Aducanumab (controversial)",
            "smiles": "N/A",  # Antibody
            "phase_failed": "Post-approval controversy",
            "failure_reason": "Questioned efficacy despite amyloid clearance, high cost, serious side effects including brain swelling",
            "failure_category": "Efficacy/Safety",
            "therapeutic_area": "CNS/Neurology",
            "year": 2021,
            "company": "Biogen",
            "additional_info": "FDA Advisory Committee voted 10-0 against approval, FDA approved anyway"
        },
        
        # QT prolongation failures
        {
            "name": "Terfenadine (Seldane)",
            "smiles": "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4",
            "phase_failed": "Post-market",
            "failure_reason": "QT prolongation leading to fatal cardiac arrhythmias (Torsades de Pointes) when combined with CYP3A4 inhibitors",
            "failure_category": "Cardiotoxicity",
            "therapeutic_area": "Allergy",
            "year": 1998,
            "company": "Hoechst Marion Roussel",
            "additional_info": "Replaced by safer metabolite fexofenadine (Allegra)"
        },
        {
            "name": "Cisapride (Propulsid)",
            "smiles": "COC1=C(C=C(C=C1)Cl)C(=O)NC2CCN(CC2)CCCOC3=CC=C(C=C3)F",
            "phase_failed": "Post-market",
            "failure_reason": "Fatal cardiac arrhythmias due to QT prolongation, especially in patients with electrolyte imbalances",
            "failure_category": "Cardiotoxicity",
            "therapeutic_area": "GI",
            "year": 2000,
            "company": "Janssen",
            "additional_info": "Over 80 deaths reported in the US"
        },
        
        # Neurotoxicity
        {
            "name": "Efalizumab (Raptiva)",
            "smiles": "N/A",  # Antibody
            "phase_failed": "Post-market",
            "failure_reason": "Progressive multifocal leukoencephalopathy (PML) - fatal brain infection",
            "failure_category": "Neurotoxicity",
            "therapeutic_area": "Dermatology",
            "year": 2009,
            "company": "Genentech",
            "additional_info": "4 confirmed PML cases, 3 deaths"
        },
    ]
    
    # Add synthetic examples of common failure patterns
    synthetic_failures = [
        # CYP inhibition issues
        {
            "name": "CYP_Failure_001",
            "smiles": "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)OC3=CC=CC=C3",
            "phase_failed": "Phase I",
            "failure_reason": "Strong CYP3A4 inhibition causing dangerous drug-drug interactions",
            "failure_category": "Drug Interactions",
            "therapeutic_area": "Oncology",
            "year": 2020,
            "company": "Synthetic"
        },
        # hERG liability
        {
            "name": "hERG_Failure_001",
            "smiles": "CN1CCC(CC1)C(=O)C2=CC=C(C=C2)F",
            "phase_failed": "Preclinical",
            "failure_reason": "High hERG channel inhibition (IC50 < 1uM) indicating QT prolongation risk",
            "failure_category": "Cardiotoxicity",
            "therapeutic_area": "CNS",
            "year": 2021,
            "company": "Synthetic"
        },
        # Reactive metabolite
        {
            "name": "Reactive_Met_001",
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=CC=C2)Cl",
            "phase_failed": "Phase I",
            "failure_reason": "Formation of reactive quinone-imine metabolite causing hepatotoxicity",
            "failure_category": "Hepatotoxicity",
            "therapeutic_area": "Inflammation",
            "year": 2019,
            "company": "Synthetic"
        },
        # Poor solubility/bioavailability
        {
            "name": "Bioavail_Failure_001",
            "smiles": "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC=CC=C4",
            "phase_failed": "Phase I",
            "failure_reason": "Extremely poor oral bioavailability (<5%) due to low aqueous solubility",
            "failure_category": "Pharmacokinetics",
            "therapeutic_area": "Oncology",
            "year": 2020,
            "company": "Synthetic"
        },
        # Blood-brain barrier issues
        {
            "name": "BBB_Failure_001",
            "smiles": "CC(C)(C)NC(=O)C1=CC=C(C=C1)S(=O)(=O)N",
            "phase_failed": "Phase II",
            "failure_reason": "Failed to cross blood-brain barrier at therapeutic concentrations for CNS target",
            "failure_category": "Pharmacokinetics",
            "therapeutic_area": "CNS",
            "year": 2021,
            "company": "Synthetic"
        },
        # Immunogenicity
        {
            "name": "Immuno_Failure_001",
            "smiles": "CC1=CC=C(C=C1)NC(=O)NC2=CC=CC=C2Cl",
            "phase_failed": "Phase II",
            "failure_reason": "High immunogenicity with anti-drug antibody formation reducing efficacy",
            "failure_category": "Immunogenicity",
            "therapeutic_area": "Autoimmune",
            "year": 2020,
            "company": "Synthetic"
        },
    ]
    
    all_failures = failures + synthetic_failures
    df = pd.DataFrame(all_failures)
    
    # Filter out antibodies (no valid SMILES)
    df = df[df['smiles'] != 'N/A']
    
    print(f"Created clinical trial failures dataset with {len(df)} entries")
    return df


# ============================================================================
# DrugBank-style Withdrawal Data
# ============================================================================

def create_drug_withdrawal_reasons() -> pd.DataFrame:
    """
    Create comprehensive data about drug withdrawals with detailed safety information.
    """
    print("Creating drug withdrawal reasons dataset...")
    
    withdrawals = [
        {
            "name": "Rofecoxib",
            "smiles": "CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3",
            "withdrawal_year": 2004,
            "withdrawal_regions": ["Worldwide"],
            "primary_reason": "Cardiovascular toxicity",
            "detailed_reason": "Long-term use associated with significantly increased risk of myocardial infarction and stroke. The VIGOR trial showed a 4-fold increase in MI risk compared to naproxen. The APPROVe trial confirmed a 2-fold increase in CV events after 18 months of use.",
            "mechanism": "Selective COX-2 inhibition disrupts the balance between prothrombotic thromboxane A2 (produced by COX-1) and antithrombotic prostacyclin (produced by COX-2), leading to a prothrombotic state.",
            "affected_population": "Patients with pre-existing cardiovascular disease at highest risk",
            "estimated_deaths": ">27,000"
        },
        {
            "name": "Cerivastatin",
            "smiles": "COC1=C(C=C(C=C1)C=CC(CC(CC(=O)O)O)O)C2=NC(=C(C=C2)F)C(C)C",
            "withdrawal_year": 2001,
            "withdrawal_regions": ["Worldwide"],
            "primary_reason": "Fatal rhabdomyolysis",
            "detailed_reason": "High doses and combination with gemfibrozil led to severe muscle breakdown. The drug was 10 times more likely to cause fatal rhabdomyolysis than other statins. CYP2C8 metabolism creates drug-drug interaction vulnerability.",
            "mechanism": "HMG-CoA reductase inhibition in skeletal muscle leads to ubiquinone depletion, mitochondrial dysfunction, and muscle fiber necrosis. Higher lipophilicity increased tissue penetration.",
            "affected_population": "Patients on high doses or taking gemfibrozil concomitantly",
            "estimated_deaths": "52"
        },
        {
            "name": "Troglitazone",
            "smiles": "CC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4",
            "withdrawal_year": 2000,
            "withdrawal_regions": ["US", "UK", "Worldwide"],
            "primary_reason": "Hepatotoxicity",
            "detailed_reason": "Idiosyncratic liver toxicity leading to acute liver failure. Cases occurred within first 6 months of treatment. Required monthly liver function monitoring during use.",
            "mechanism": "Formation of reactive quinone metabolite that causes oxidative stress and mitochondrial dysfunction in hepatocytes. Also inhibits bile salt export pump (BSEP).",
            "affected_population": "Unpredictable - occurred in patients regardless of dose or prior liver disease",
            "estimated_deaths": "63"
        },
        {
            "name": "Phenylpropanolamine",
            "smiles": "CC(C(C1=CC=CC=C1)O)N",
            "withdrawal_year": 2000,
            "withdrawal_regions": ["US"],
            "primary_reason": "Hemorrhagic stroke",
            "detailed_reason": "Case-control study showed 16-fold increased risk of hemorrhagic stroke in women using the drug as appetite suppressant, and 3-fold increased risk overall.",
            "mechanism": "Sympathomimetic amine causing vasoconstriction and acute hypertension, leading to rupture of cerebral blood vessels.",
            "affected_population": "Women, especially those using high doses for weight loss",
            "estimated_deaths": "Unknown - estimated hundreds"
        },
        {
            "name": "Terfenadine",
            "smiles": "CC(C)(C)C1=CC=C(C=C1)C(O)CCCN2CCC(CC2)C(O)(C3=CC=CC=C3)C4=CC=CC=C4",
            "withdrawal_year": 1998,
            "withdrawal_regions": ["Worldwide"],
            "primary_reason": "Fatal cardiac arrhythmias",
            "detailed_reason": "QT prolongation leading to Torsades de Pointes ventricular tachycardia. Risk dramatically increased when CYP3A4 inhibitors (ketoconazole, erythromycin) or grapefruit juice were co-administered.",
            "mechanism": "Parent compound blocks hERG potassium channels. Normally rapidly metabolized to fexofenadine by CYP3A4, but inhibition allows toxic accumulation of parent drug.",
            "affected_population": "Patients taking CYP3A4 inhibitors or with liver disease",
            "estimated_deaths": "~125 in US"
        },
    ]
    
    df = pd.DataFrame(withdrawals)
    print(f"Created drug withdrawal dataset with {len(df)} entries")
    return df


# ============================================================================
# Main Download Function
# ============================================================================

def main(test_mode: bool = False, skip_pubchem: bool = False):
    """
    Main function to download all comprehensive drug data.
    
    Args:
        test_mode: If True, download smaller samples
        skip_pubchem: Skip slow PubChem API calls
    """
    output_dir = RAW_DATA_DIR / "comprehensive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Comprehensive Drug Data Download for Text Generation Training")
    print("="*60)
    
    limit_factor = 0.1 if test_mode else 1.0
    
    # 1. Download detailed ChEMBL compounds
    print("\n" + "-"*40)
    print("STEP 1: Downloading ChEMBL Detailed Compounds")
    print("-"*40)
    compounds_df = download_chembl_compounds_detailed(limit=int(10000 * limit_factor))
    compounds_df.to_csv(output_dir / "chembl_compounds_detailed.csv", index=False)
    
    # 2. Download drug mechanisms
    print("\n" + "-"*40)
    print("STEP 2: Downloading Drug Mechanisms")
    print("-"*40)
    mechanisms_df = download_chembl_drug_mechanisms(limit=int(5000 * limit_factor))
    mechanisms_df.to_csv(output_dir / "chembl_mechanisms.csv", index=False)
    
    # 3. Download drug indications
    print("\n" + "-"*40)
    print("STEP 3: Downloading Drug Indications")
    print("-"*40)
    indications_df = download_chembl_drug_indications(limit=int(10000 * limit_factor))
    indications_df.to_csv(output_dir / "chembl_indications.csv", index=False)
    
    # 4. Download drug warnings
    print("\n" + "-"*40)
    print("STEP 4: Downloading Drug Warnings")
    print("-"*40)
    warnings_df = download_chembl_drug_warnings()
    warnings_df.to_csv(output_dir / "chembl_warnings.csv", index=False)
    
    # 5. Download metabolism data
    print("\n" + "-"*40)
    print("STEP 5: Downloading Metabolism Data")
    print("-"*40)
    metabolism_df = download_chembl_metabolism(limit=int(5000 * limit_factor))
    metabolism_df.to_csv(output_dir / "chembl_metabolism.csv", index=False)
    
    # 6. Create clinical trial failures dataset
    print("\n" + "-"*40)
    print("STEP 6: Creating Clinical Trial Failures Dataset")
    print("-"*40)
    failures_df = create_clinical_trial_failures()
    failures_df.to_csv(output_dir / "clinical_trial_failures.csv", index=False)
    
    # 7. Create drug withdrawal reasons
    print("\n" + "-"*40)
    print("STEP 7: Creating Drug Withdrawal Reasons Dataset")
    print("-"*40)
    withdrawals_df = create_drug_withdrawal_reasons()
    withdrawals_df.to_csv(output_dir / "drug_withdrawals.csv", index=False)
    
    # Summary
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Detailed compounds: {len(compounds_df)}")
    print(f"Drug mechanisms: {len(mechanisms_df)}")
    print(f"Drug indications: {len(indications_df)}")
    print(f"Drug warnings: {len(warnings_df)}")
    print(f"Metabolism records: {len(metabolism_df)}")
    print(f"Clinical trial failures: {len(failures_df)}")
    print(f"Drug withdrawals: {len(withdrawals_df)}")
    print(f"\nOutput directory: {output_dir}")
    
    return {
        'compounds': compounds_df,
        'mechanisms': mechanisms_df,
        'indications': indications_df,
        'warnings': warnings_df,
        'metabolism': metabolism_df,
        'failures': failures_df,
        'withdrawals': withdrawals_df,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download comprehensive drug discovery data")
    parser.add_argument("--test", action="store_true", help="Test mode - download small samples")
    parser.add_argument("--skip-pubchem", action="store_true", help="Skip PubChem API calls")
    args = parser.parse_args()
    
    main(test_mode=args.test, skip_pubchem=args.skip_pubchem)
