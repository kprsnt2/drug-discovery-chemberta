"""
Script to update HuggingFace model card with benchmark results.
Reads from results/benchmark_results.json and updates the README.md

Usage:
    python cloud/update_model_card.py
"""

import json
from pathlib import Path
from huggingface_hub import HfApi

# Configuration
MODEL_DIR = Path("checkpoints/cloud_Qwen_2.5_14B_20251228_1607/final_model")
RESULTS_FILE = Path("results/benchmark_results.json")
EVAL_RESULTS_FILE = Path("results/evaluation_metrics.json")
REPORT_FILE = Path("results/benchmark_report.html")
REPO_ID = "kprsnt2/drug-discovery-qwen-14b"


def load_results():
    """Load benchmark and evaluation results."""
    results = {}
    
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results['benchmark'] = json.load(f)
    
    if EVAL_RESULTS_FILE.exists():
        with open(EVAL_RESULTS_FILE) as f:
            results['evaluation'] = json.load(f)
    
    return results


def generate_model_card(results):
    """Generate model card markdown with results."""
    
    # Get metrics
    ft = results.get('benchmark', {}).get('finetuned', {})
    ev = results.get('evaluation', {})
    
    # Use evaluation results if available, else benchmark
    accuracy = ev.get('accuracy', ft.get('accuracy', 'N/A'))
    f1 = ev.get('f1_score', ft.get('f1_score', 'N/A'))
    roc_auc = ev.get('roc_auc', ft.get('roc_auc', 'N/A'))
    pr_auc = ev.get('pr_auc', ft.get('pr_auc', 'N/A'))
    precision = ev.get('precision', 'N/A')
    recall = ev.get('recall', 'N/A')
    
    # Format as percentage strings
    def fmt(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)
    
    model_card = f"""---
license: apache-2.0
language:
- en
tags:
- drug-discovery
- chemistry
- molecular-property-prediction
- qwen2
- finetuned
datasets:
- custom
metrics:
- accuracy
- f1
- roc_auc
pipeline_tag: text-classification
library_name: transformers
model-index:
- name: drug-discovery-qwen-14b
  results:
  - task:
      type: text-classification
      name: Drug Approval Prediction
    metrics:
    - type: accuracy
      value: {accuracy}
    - type: f1
      value: {f1}
    - type: roc_auc
      value: {roc_auc}
---

# Drug Discovery Model - Qwen 2.5 14B Finetuned

This model is a fine-tuned version of **Qwen 2.5 14B** for **drug discovery and molecular property prediction**. It predicts whether a drug compound (represented as a SMILES string) is likely to be approved or fail in clinical trials.

## Model Description

- **Base Model:** Qwen/Qwen2.5-14B
- **Task:** Binary classification (Drug Approval Prediction)
- **Training Hardware:** AMD MI300X 192GB GPU
- **Training Framework:** HuggingFace Transformers with 4-bit quantization

## Evaluation Results

| Metric | Score |
|--------|-------|
| **Accuracy** | {fmt(accuracy)} |
| **F1 Score** | {fmt(f1)} |
| **Precision** | {fmt(precision)} |
| **Recall** | {fmt(recall)} |
| **ROC-AUC** | {fmt(roc_auc)} |
| **PR-AUC** | {fmt(pr_auc)} |

## Training Data

The model was trained on a curated dataset of drug molecules with known clinical trial outcomes:
- **Format:** SMILES molecular representations
- **Labels:** Binary (0 = Failed, 1 = Approved)

## Training Hyperparameters

- **Epochs:** 3
- **Batch Size:** 1 (with gradient accumulation)
- **Learning Rate:** 1e-5
- **Optimizer:** AdamW
- **Precision:** bfloat16 with 4-bit quantization (NF4)
- **Gradient Checkpointing:** Enabled

## How to Use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "kprsnt2/drug-discovery-qwen-14b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example SMILES string (Aspirin)
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Tokenize and predict
inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

print(f"Prediction: {{'Approved' if prediction == 1 else 'Failed'}}")
print(f"Confidence: {{probs[0][prediction].item():.2%}}")
```

## Limitations

- Model predictions should not replace expert judgment in drug development
- Performance may vary on molecules very different from training data
- SMILES representation may not capture all relevant molecular properties

## Citation

```bibtex
@misc{{drug-discovery-qwen-14b,
  author = {{Prashanth Kumar}},
  title = {{Drug Discovery Model - Qwen 2.5 14B Finetuned}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/kprsnt2/drug-discovery-qwen-14b}}
}}
```
"""
    return model_card


def main():
    print("=" * 50)
    print("Updating HuggingFace Model Card")
    print("=" * 50)
    
    # Load results
    print("\n[1/4] Loading results...")
    results = load_results()
    
    if not results:
        print("ERROR: No results files found!")
        print(f"  Looking for: {RESULTS_FILE} or {EVAL_RESULTS_FILE}")
        return
    
    print(f"  Found results: {list(results.keys())}")
    
    # Generate model card
    print("\n[2/4] Generating model card...")
    model_card = generate_model_card(results)
    
    # Save locally
    readme_path = MODEL_DIR / "README.md"
    print(f"\n[3/4] Saving to {readme_path}...")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print("  ✓ Saved locally")
    
    # Upload to HuggingFace
    print(f"\n[4/4] Uploading to HuggingFace ({REPO_ID})...")
    try:
        api = HfApi()
        
        # Upload README
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("  ✓ README.md uploaded")
        
        # Upload HTML report if exists
        if REPORT_FILE.exists():
            api.upload_file(
                path_or_fileobj=str(REPORT_FILE),
                path_in_repo="benchmark_report.html",
                repo_id=REPO_ID,
                repo_type="model"
            )
            print("  ✓ benchmark_report.html uploaded")
        
        # Upload results JSON
        if EVAL_RESULTS_FILE.exists():
            api.upload_file(
                path_or_fileobj=str(EVAL_RESULTS_FILE),
                path_in_repo="evaluation_metrics.json",
                repo_id=REPO_ID,
                repo_type="model"
            )
            print("  ✓ evaluation_metrics.json uploaded")
        
        print("\n" + "=" * 50)
        print("✅ Model card updated successfully!")
        print(f"View at: https://huggingface.co/{REPO_ID}")
        print("=" * 50)
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Model card saved locally but not uploaded.")


if __name__ == "__main__":
    main()
