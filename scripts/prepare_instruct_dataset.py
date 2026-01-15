"""
Instruction-Tuning Dataset Preparation for Drug Discovery LLM

Converts raw drug data into instruction-tuning format with:
- Drug analysis prompts with explanatory responses
- Comparison tasks
- Failure reason explanations
- Safety recommendations
- Molecular property analysis
"""

import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ============================================================================
# Prompt Templates
# ============================================================================

ANALYSIS_TEMPLATES = [
    "Analyze this drug candidate and predict its approval likelihood:\nSMILES: {smiles}\nDrug Name: {name}",
    "Evaluate the safety and efficacy potential of the following molecule:\n{smiles}\nName: {name}",
    "Provide a comprehensive drug development assessment for:\nMolecular Structure: {smiles}\nCandidate Name: {name}",
    "As a pharmaceutical researcher, analyze this compound:\n{smiles}\n{name}",
    "Predict whether this drug candidate will succeed in clinical trials:\n{smiles}",
]

COMPARISON_TEMPLATES = [
    "Compare the safety profiles of these two drugs:\nDrug 1: {smiles1} ({name1})\nDrug 2: {smiles2} ({name2})",
    "Which drug candidate has better development potential?\nA: {smiles1} ({name1})\nB: {smiles2} ({name2})",
    "Analyze the differences between these molecules and predict which is more likely to succeed:\n1. {name1}: {smiles1}\n2. {name2}: {smiles2}",
]

FAILURE_ANALYSIS_TEMPLATES = [
    "This drug failed in {phase}. Explain why:\nDrug: {name}\nSMILES: {smiles}",
    "Analyze why {name} was withdrawn/failed and what lessons can be learned:\nStructure: {smiles}",
    "What structural or pharmacological properties led to the failure of {name}?\n{smiles}",
]

IMPROVEMENT_TEMPLATES = [
    "This drug failed due to {reason}. Suggest structural modifications to improve safety:\n{smiles}",
    "How could {name} be modified to reduce its toxicity risks?\nOriginal structure: {smiles}",
    "Propose a safer analog of this failed drug:\n{name}: {smiles}\nFailure reason: {reason}",
]

PROPERTY_TEMPLATES = [
    "Describe the molecular properties and drug-likeness of:\n{smiles}",
    "Analyze the ADMET properties of this compound:\n{smiles}",
    "Evaluate the Lipinski's Rule of 5 compliance and pharmacokinetic potential:\n{smiles}",
]


# ============================================================================
# Response Generation
# ============================================================================

def generate_approval_prediction_response(row: pd.Series) -> str:
    """Generate detailed approval prediction response."""
    
    status = row.get('status', 'unknown')
    max_phase = row.get('max_phase', 0)
    name = row.get('name', 'Unknown compound')
    smiles = row.get('smiles', '')
    
    # Calculate drug-likeness indicators
    mw = row.get('molecular_weight', 0) or 0
    alogp = row.get('alogp', 0) or 0
    hba = row.get('hba', 0) or 0
    hbd = row.get('hbd', 0) or 0
    psa = row.get('psa', 0) or 0
    ro5_violations = row.get('num_ro5_violations', 0) or 0
    qed = row.get('qed_weighted', 0) or 0
    
    # Determine prediction
    if status == 'approved' or max_phase == 4:
        prediction = "APPROVED"
        confidence = "High"
        emoji = "✅"
    elif status in ['failed', 'withdrawn']:
        prediction = "FAILED/WITHDRAWN"
        confidence = "High"
        emoji = "❌"
    elif max_phase >= 2:
        prediction = "IN DEVELOPMENT"
        confidence = "Medium"
        emoji = "🔬"
    else:
        prediction = "UNLIKELY TO SUCCEED"
        confidence = "Medium"
        emoji = "⚠️"
    
    # Build response
    response = f"""## Drug Analysis Report

### Prediction: {emoji} {prediction}
**Confidence Level:** {confidence}

### Molecular Properties Analysis

| Property | Value | Assessment |
|----------|-------|------------|
| Molecular Weight | {mw:.1f} Da | {"✓ Good (<500)" if mw < 500 else "⚠ High (>500)"} |
| LogP (Lipophilicity) | {alogp:.2f} | {"✓ Good (<5)" if alogp < 5 else "⚠ High (>5)"} |
| H-Bond Acceptors | {hba} | {"✓ Good (≤10)" if hba <= 10 else "⚠ High (>10)"} |
| H-Bond Donors | {hbd} | {"✓ Good (≤5)" if hbd <= 5 else "⚠ High (>5)"} |
| Polar Surface Area | {psa:.1f} Å² | {"✓ Good (≤140)" if psa <= 140 else "⚠ High (>140)"} |
| Rule of 5 Violations | {ro5_violations} | {"✓ Compliant" if ro5_violations == 0 else "⚠ Non-compliant"} |

### Drug-Likeness Assessment
"""
    
    # Add drug-likeness reasoning
    if ro5_violations == 0 and mw < 500 and alogp < 5:
        response += """
The compound shows **excellent drug-like properties**:
- Fully compliant with Lipinski's Rule of 5
- Good balance of hydrophilicity/lipophilicity for membrane permeation
- Suitable molecular weight for oral bioavailability
"""
    elif ro5_violations <= 1:
        response += """
The compound shows **acceptable drug-like properties** with minor concerns:
- Minor deviation from Lipinski's Rule of 5
- May require formulation optimization for oral delivery
"""
    else:
        response += """
The compound shows **poor drug-like properties**:
- Multiple violations of Lipinski's Rule of 5
- May have bioavailability challenges
- Consider structural modifications to improve pharmacokinetics
"""
    
    # Add specific guidance based on status
    if status in ['failed', 'withdrawn']:
        failure_reason = row.get('failure_reason', row.get('warning_type', 'Safety concerns'))
        response += f"""
### Failure Analysis

**Reason for Failure:** {failure_reason}

This compound was withdrawn/failed likely due to:
1. **Safety signals** detected in clinical trials or post-market surveillance
2. **Mechanism-related toxicity** affecting non-target tissues
3. **Drug-drug interaction** potential from metabolic vulnerabilities

### Lessons Learned
To avoid similar failures, future candidates should:
- Undergo comprehensive safety profiling early in development
- Include diversity in clinical trial populations
- Monitor for class-related adverse effects
"""
    elif status == 'approved':
        response += f"""
### Success Factors

This drug was successfully approved, demonstrating:
1. **Favorable safety profile** through clinical development
2. **Clear efficacy** in target patient population
3. **Acceptable risk-benefit ratio** for intended indication

### Key Success Indicators
- Completed Phase III trials with positive outcomes
- Manageable side effect profile
- Clear differentiation from existing treatments
"""
    
    return response.strip()


def generate_failure_analysis_response(row: pd.Series) -> str:
    """Generate detailed failure analysis response."""
    
    name = row.get('name', 'Unknown')
    smiles = row.get('smiles', '')
    phase = row.get('phase_failed', 'Unknown phase')
    reason = row.get('failure_reason', row.get('detailed_reason', 'Safety/Efficacy concerns'))
    category = row.get('failure_category', 'Unknown')
    mechanism = row.get('mechanism', '')
    therapeutic_area = row.get('therapeutic_area', '')
    year = row.get('year', row.get('withdrawal_year', ''))
    
    response = f"""## Drug Failure Analysis: {name}

### Summary
**Phase Failed:** {phase}
**Year:** {year}
**Therapeutic Area:** {therapeutic_area}
**Failure Category:** {category}

### Detailed Failure Reason

{reason}
"""
    
    if mechanism:
        response += f"""
### Mechanism of Toxicity

{mechanism}
"""
    
    # Add category-specific analysis
    category_lower = str(category).lower()
    
    if 'cardio' in category_lower or 'cardiac' in category_lower:
        response += """
### Cardiovascular Toxicity Analysis

**Key Risk Factors:**
1. **hERG Channel Inhibition:** Many cardiotoxic drugs block the hERG potassium channel, leading to QT prolongation and potentially fatal arrhythmias (Torsades de Pointes)
2. **Prothrombotic Effects:** Selective COX-2 inhibitors reduce prostacyclin production, tipping the balance toward platelet aggregation
3. **Myocardial Stress:** Some drugs cause direct cardiomyocyte damage or increase cardiac workload

**Structural Alerts:**
- Tertiary amines with lipophilic aromatic rings often show hERG liability
- Basic nitrogen at appropriate distance from aromatic system increases hERG risk

**Prevention Strategies:**
- Early hERG screening (patch clamp assays)
- In vivo QT studies in multiple species
- Careful monitoring in clinical trials for PR/QT changes
"""
    elif 'hepato' in category_lower or 'liver' in category_lower:
        response += """
### Hepatotoxicity Analysis

**Key Risk Factors:**
1. **Reactive Metabolite Formation:** CYP450 metabolism can generate electrophilic species that bind to proteins
2. **Bile Salt Transport Inhibition:** BSEP inhibition leads to intrahepatic cholestasis
3. **Mitochondrial Toxicity:** Disruption of oxidative phosphorylation in hepatocytes

**Structural Alerts:**
- Quinone/quinone-imine forming structures
- Michael acceptors (α,β-unsaturated carbonyls)
- Compounds with furan, thiophene rings
- Hydrazines and anilines

**Prevention Strategies:**
- Reactive metabolite trapping studies
- BSEP inhibition assays
- Mitochondrial toxicity screening
- In vivo liver function biomarker monitoring
"""
    elif 'efficacy' in category_lower:
        response += """
### Efficacy Failure Analysis

**Common Reasons for Efficacy Failure:**
1. **Target Selection Issues:** Pre-clinical models may not accurately reflect human disease biology
2. **Patient Population Heterogeneity:** Subgroups may respond differently
3. **Dose Selection:** May not achieve adequate target engagement
4. **Biomarker Disconnect:** Surrogate endpoints may not correlate with clinical outcomes

**Prevention Strategies:**
- Extensive target validation using human genetics data
- Biomarker-driven patient stratification
- PK/PD modeling to optimize dosing
- Adaptive trial designs
"""
    
    response += """
### Key Takeaways for Drug Development

1. **Early Safety Screening:** Implement comprehensive in vitro and in vivo safety panels before clinical development
2. **Mechanism Understanding:** Deep mechanistic understanding helps predict and mitigate risks
3. **Patient Monitoring:** Identify high-risk patient populations and implement appropriate monitoring
4. **Structural Optimization:** Use SAR to move away from structural liabilities while maintaining efficacy
"""
    
    return response.strip()


def generate_comparison_response(drug1: pd.Series, drug2: pd.Series) -> str:
    """Generate drug comparison response."""
    
    name1 = drug1.get('name', 'Drug 1')
    name2 = drug2.get('name', 'Drug 2')
    status1 = drug1.get('status', 'unknown')
    status2 = drug2.get('status', 'unknown')
    
    mw1, mw2 = drug1.get('molecular_weight', 0) or 0, drug2.get('molecular_weight', 0) or 0
    alogp1, alogp2 = drug1.get('alogp', 0) or 0, drug2.get('alogp', 0) or 0
    psa1, psa2 = drug1.get('psa', 0) or 0, drug2.get('psa', 0) or 0
    
    response = f"""## Comparative Drug Analysis

### Head-to-Head Comparison

| Property | {name1} | {name2} | Better |
|----------|---------|---------|--------|
| Status | {status1} | {status2} | {"🏆 " + name1 if status1 == 'approved' else "🏆 " + name2 if status2 == 'approved' else "—"} |
| Mol. Weight | {mw1:.1f} Da | {mw2:.1f} Da | {"🏆 " + name1 if mw1 < mw2 else "🏆 " + name2} |
| LogP | {alogp1:.2f} | {alogp2:.2f} | {"🏆 " + name1 if abs(alogp1 - 2) < abs(alogp2 - 2) else "🏆 " + name2} |
| PSA | {psa1:.1f} Å² | {psa2:.1f} Å² | {"🏆 " + name1 if psa1 < psa2 else "🏆 " + name2} |

### Assessment

"""
    
    if status1 == 'approved' and status2 != 'approved':
        response += f"""**{name1}** is the clear winner as it has achieved regulatory approval, demonstrating successful clinical development with an acceptable safety/efficacy profile.

**{name2}** has not achieved approval, which may indicate:
- Efficacy challenges in clinical trials
- Safety concerns that limited development
- Strategic decision to discontinue development
"""
    elif status2 == 'approved' and status1 != 'approved':
        response += f"""**{name2}** is the clear winner as it has achieved regulatory approval, demonstrating successful clinical development with an acceptable safety/efficacy profile.

**{name1}** has not achieved approval, which may indicate:
- Efficacy challenges in clinical trials
- Safety concerns that limited development
- Strategic decision to discontinue development
"""
    else:
        response += f"""Both compounds are at similar development stages. Based on molecular properties:

- **{name1}** may have {"better" if mw1 < 500 and alogp1 < 5 else "challenging"} oral bioavailability
- **{name2}** may have {"better" if mw2 < 500 and alogp2 < 5 else "challenging"} oral bioavailability

Further investigation into mechanism of action, selectivity, and target engagement would be needed for definitive comparison.
"""
    
    return response.strip()


def generate_property_response(row: pd.Series) -> str:
    """Generate molecular property analysis response."""
    
    smiles = row.get('smiles', '')
    name = row.get('name', 'Unknown')
    mw = row.get('molecular_weight', 0) or 0
    alogp = row.get('alogp', 0) or 0
    hba = row.get('hba', 0) or 0
    hbd = row.get('hbd', 0) or 0
    psa = row.get('psa', 0) or 0
    rtb = row.get('rtb', 0) or 0
    qed = row.get('qed_weighted', 0) or 0
    natural = row.get('natural_product', False)
    
    response = f"""## Molecular Property Analysis: {name}

### Structure
```
{smiles}
```

### Physicochemical Properties

| Property | Value | Optimal Range | Status |
|----------|-------|---------------|--------|
| Molecular Weight | {mw:.1f} Da | 150-500 Da | {"✓" if 150 <= mw <= 500 else "⚠"} |
| LogP (cLogP) | {alogp:.2f} | 0-5 | {"✓" if 0 <= alogp <= 5 else "⚠"} |
| H-Bond Acceptors | {hba} | ≤10 | {"✓" if hba <= 10 else "⚠"} |
| H-Bond Donors | {hbd} | ≤5 | {"✓" if hbd <= 5 else "⚠"} |
| PSA | {psa:.1f} Å² | <140 Å² | {"✓" if psa < 140 else "⚠"} |
| Rotatable Bonds | {rtb} | ≤10 | {"✓" if rtb <= 10 else "⚠"} |

### Drug-Likeness Score
**QED (Quantitative Estimate of Drug-likeness):** {qed:.3f}
- Score ranges from 0 (least drug-like) to 1 (most drug-like)
- QED >0.5 is generally considered favorable

### ADMET Predictions

**Absorption:**
- {"Good oral absorption expected" if psa < 140 and alogp < 5 else "May have oral absorption challenges"}
- {"Likely CNS penetrant" if psa < 90 and 1 < alogp < 5 else "Limited CNS penetration expected"}

**Distribution:**
- {"High plasma protein binding likely" if alogp > 3 else "Moderate plasma protein binding expected"}
- Estimated Vd: {"High (lipophilic)" if alogp > 3 else "Moderate" if alogp > 1 else "Low (hydrophilic)"}

**Metabolism:**
- {"CYP450 substrate likely" if mw > 300 and alogp > 1 else "May undergo phase II metabolism primarily"}
- {"Potential CYP inhibition risk" if alogp > 4 else "Low CYP inhibition expected"}

**Excretion:**
- {"Likely hepatic clearance" if alogp > 1 else "May undergo renal clearance"}
- {"Long half-life expected" if mw > 400 and alogp > 3 else "Moderate half-life expected"}

### Rule-Based Assessments

**Lipinski's Rule of 5:** {"✓ PASS" if mw <= 500 and alogp <= 5 and hbd <= 5 and hba <= 10 else "⚠ FAIL"}
**Veber Rules (Oral Bioavailability):** {"✓ PASS" if rtb <= 10 and psa < 140 else "⚠ FAIL"}
"""
    
    if natural:
        response += """
### Natural Product Flag
This compound is derived from or inspired by natural products, which often have:
- Complex stereochemistry
- Unique mechanisms of action
- Potentially better target selectivity
"""
    
    return response.strip()


# ============================================================================
# Dataset Generation
# ============================================================================

def create_instruction_tuning_dataset(output_dir: Path, max_samples: int = None) -> pd.DataFrame:
    """
    Create comprehensive instruction-tuning dataset from all available drug data.
    """
    print("Creating instruction-tuning dataset...")
    
    data_sources = []
    
    # Load comprehensive data if available
    comprehensive_dir = RAW_DATA_DIR / "comprehensive"
    if comprehensive_dir.exists():
        for csv_file in comprehensive_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['source'] = csv_file.stem
                data_sources.append(df)
                print(f"  Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
    
    # Load processed data
    processed_dir = PROCESSED_DATA_DIR
    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
        csv_path = processed_dir / csv_file
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df['source'] = 'processed'
                data_sources.append(df)
                print(f"  Loaded {len(df)} records from {csv_file}")
            except Exception as e:
                print(f"  Error loading {csv_path}: {e}")
    
    if not data_sources:
        print("No data sources found! Please run download scripts first.")
        return pd.DataFrame()
    
    # Combine all data
    all_data = pd.concat(data_sources, ignore_index=True)
    print(f"\nTotal records: {len(all_data)}")
    
    # Deduplicate by SMILES
    if 'smiles' in all_data.columns:
        all_data = all_data.drop_duplicates(subset='smiles', keep='first')
        print(f"After deduplication: {len(all_data)}")
    
    # Generate instruction-tuning samples
    samples = []
    
    # 1. Drug Analysis Samples
    print("\nGenerating drug analysis samples...")
    compounds_with_data = all_data[all_data['smiles'].notna() & (all_data['smiles'] != '')]
    
    for idx, row in tqdm(compounds_with_data.iterrows(), total=len(compounds_with_data), desc="Drug Analysis"):
        if max_samples and len(samples) >= max_samples:
            break
        
        template = random.choice(ANALYSIS_TEMPLATES)
        instruction = template.format(
            smiles=row.get('smiles', ''),
            name=row.get('name', 'Unknown')
        )
        response = generate_approval_prediction_response(row)
        
        samples.append({
            'instruction': instruction,
            'input': '',
            'output': response,
            'task_type': 'drug_analysis',
            'source': row.get('source', 'unknown')
        })
    
    # 2. Failure Analysis Samples
    print("\nGenerating failure analysis samples...")
    failures = all_data[all_data['source'].isin(['clinical_trial_failures', 'drug_withdrawals'])]
    
    for idx, row in tqdm(failures.iterrows(), total=len(failures), desc="Failure Analysis"):
        template = random.choice(FAILURE_ANALYSIS_TEMPLATES)
        instruction = template.format(
            name=row.get('name', 'Unknown'),
            smiles=row.get('smiles', ''),
            phase=row.get('phase_failed', 'clinical trials')
        )
        response = generate_failure_analysis_response(row)
        
        samples.append({
            'instruction': instruction,
            'input': '',
            'output': response,
            'task_type': 'failure_analysis',
            'source': row.get('source', 'unknown')
        })
    
    # 3. Improvement Suggestion Samples
    print("\nGenerating improvement suggestion samples...")
    failed_drugs = all_data[all_data['status'].isin(['failed', 'withdrawn']) | all_data['failure_reason'].notna()]
    
    for idx, row in tqdm(failed_drugs.head(500).iterrows(), total=min(500, len(failed_drugs)), desc="Improvements"):
        template = random.choice(IMPROVEMENT_TEMPLATES)
        instruction = template.format(
            name=row.get('name', 'Unknown'),
            smiles=row.get('smiles', ''),
            reason=row.get('failure_reason', row.get('warning_type', 'toxicity'))
        )
        
        response = f"""## Structural Improvement Suggestions for {row.get('name', 'Unknown')}

### Original Issue
{row.get('failure_reason', row.get('warning_type', 'Safety concerns'))}

### Recommended Modifications

1. **Reduce Metabolic Liability**
   - Replace metabolically labile groups (e.g., ester → amide)
   - Block sites of CYP450 oxidation with fluorine substitution
   - Reduce lipophilicity to minimize reactive metabolite formation

2. **Improve Selectivity**
   - Add stereochemical constraints to reduce off-target binding
   - Modify substituents to exploit differences in target binding sites
   - Use structure-guided design based on target crystal structures

3. **Address Specific Toxicity**
   - If cardiotoxicity: reduce basicity and lipophilicity to lower hERG risk
   - If hepatotoxicity: remove metabolically activated groups
   - If CNS effects: increase polarity to reduce brain penetration

### General Principles
- Maintain target engagement (measure IC50/EC50)
- Improve therapeutic index (efficacy dose vs toxicity dose)
- Consider prodrug strategies for problematic functional groups
"""
        
        samples.append({
            'instruction': instruction,
            'input': '',
            'output': response,
            'task_type': 'improvement_suggestion',
            'source': row.get('source', 'unknown')
        })
    
    # 4. Property Analysis Samples
    print("\nGenerating property analysis samples...")
    compounds_with_props = all_data[all_data['molecular_weight'].notna()]
    
    for idx, row in tqdm(compounds_with_props.head(1000).iterrows(), total=min(1000, len(compounds_with_props)), desc="Properties"):
        template = random.choice(PROPERTY_TEMPLATES)
        instruction = template.format(smiles=row.get('smiles', ''))
        response = generate_property_response(row)
        
        samples.append({
            'instruction': instruction,
            'input': '',
            'output': response,
            'task_type': 'property_analysis',
            'source': row.get('source', 'unknown')
        })
    
    # 5. Comparison Samples
    print("\nGenerating comparison samples...")
    approved = all_data[all_data['status'] == 'approved'].head(200)
    failed = all_data[all_data['status'].isin(['failed', 'withdrawn'])].head(200)
    
    for i in tqdm(range(min(200, len(approved), len(failed))), desc="Comparisons"):
        drug1 = approved.iloc[i]
        drug2 = failed.iloc[i]
        
        template = random.choice(COMPARISON_TEMPLATES)
        instruction = template.format(
            smiles1=drug1.get('smiles', ''),
            name1=drug1.get('name', 'Drug A'),
            smiles2=drug2.get('smiles', ''),
            name2=drug2.get('name', 'Drug B')
        )
        response = generate_comparison_response(drug1, drug2)
        
        samples.append({
            'instruction': instruction,
            'input': '',
            'output': response,
            'task_type': 'comparison',
            'source': 'generated'
        })
    
    # Create final dataset
    dataset = pd.DataFrame(samples)
    
    # Shuffle
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create text column for training
    dataset['text'] = dataset.apply(
        lambda row: f"""<|im_start|>user
{row['instruction']}{' ' + row['input'] if row['input'] else ''}
<|im_end|>
<|im_start|>assistant
{row['output']}
<|im_end|>""",
        axis=1
    )
    
    # Split into train/val/test
    n = len(dataset)
    train_end = int(0.85 * n)
    val_end = int(0.95 * n)
    
    train_df = dataset.iloc[:train_end]
    val_df = dataset.iloc[train_end:val_end]
    test_df = dataset.iloc[val_end:]
    
    # Save datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train_instruct.csv", index=False)
    val_df.to_csv(output_dir / "val_instruct.csv", index=False)
    test_df.to_csv(output_dir / "test_instruct.csv", index=False)
    
    # Save as JSONL for training
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        with open(output_dir / f"{split_name}_instruct.jsonl", 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                f.write(json.dumps({
                    'text': row['text'],
                    'instruction': row['instruction'],
                    'output': row['output'],
                    'task_type': row['task_type'],
                }, ensure_ascii=False) + '\n')
    
    print("\n" + "="*60)
    print("INSTRUCTION-TUNING DATASET CREATED!")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"  Training: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    print(f"\nTask distribution:")
    print(dataset['task_type'].value_counts().to_string())
    print(f"\nOutput directory: {output_dir}")
    
    return dataset


def main():
    """Main function to create instruction-tuning dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create instruction-tuning dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per category")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR / "instruct"
    
    create_instruction_tuning_dataset(output_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
