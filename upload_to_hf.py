"""
Upload Model to Hugging Face Hub

Publishes the finetuned drug discovery model to Hugging Face.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from config import MODEL_CONFIG, CHECKPOINT_DIR, RESULTS_DIR
from src.model import DrugDiscoveryModel, get_tokenizer

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import HfApi, create_repo, upload_folder


def prepare_model_for_upload(checkpoint_path: str, output_dir: str):
    """
    Prepare model files for Hugging Face upload.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Directory to save prepared files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading trained model...")
    model = DrugDiscoveryModel(use_gradient_checkpointing=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save model in HuggingFace format
    print("Saving model weights...")
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    
    # Save config
    print("Saving config...")
    config_dict = {
        "model_type": "roberta",
        "base_model": MODEL_CONFIG['model_name'],
        "num_labels": MODEL_CONFIG['num_labels'],
        "max_length": MODEL_CONFIG['max_length'],
        "task": "drug-success-prediction",
        "labels": {
            "0": "failed",
            "1": "approved"
        }
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Copy tokenizer
    print("Saving tokenizer...")
    tokenizer = get_tokenizer()
    tokenizer.save_pretrained(output_dir)
    
    # Copy benchmark results if available
    benchmark_path = RESULTS_DIR / "benchmark_results.json"
    if benchmark_path.exists():
        import shutil
        shutil.copy(benchmark_path, output_dir / "benchmark_results.json")
    
    print(f"Model prepared in: {output_dir}")
    return output_dir


def create_model_card(output_dir: str, metrics: dict = None):
    """Create README.md model card for Hugging Face."""
    
    # Load metrics if available
    if metrics is None:
        benchmark_path = RESULTS_DIR / "benchmark_results.json"
        if benchmark_path.exists():
            with open(benchmark_path, 'r') as f:
                all_metrics = json.load(f)
                metrics = all_metrics.get('finetuned', {})
    
    accuracy = metrics.get('accuracy', 0) * 100 if metrics else 0
    f1 = metrics.get('f1_score', 0) * 100 if metrics else 0
    roc_auc = metrics.get('roc_auc', 0) * 100 if metrics else 0
    pr_auc = metrics.get('pr_auc', 0) * 100 if metrics else 0
    
    model_card = f'''---
language: en
license: mit
tags:
  - drug-discovery
  - chemberta
  - smiles
  - molecular-property-prediction
  - pharmaceuticals
  - chemistry
datasets:
  - custom
metrics:
  - accuracy
  - f1
  - roc_auc
pipeline_tag: text-classification
---

# Drug Discovery Model - ChemBERTa Finetuned

A ChemBERTa model finetuned for predicting drug approval success based on molecular SMILES representation.

## Model Description

This model is a finetuned version of [seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) for drug success prediction.
It classifies molecules as likely to be **approved** or **failed** based on their SMILES structure.

### Training Details

- **Base Model**: ChemBERTa-zinc-base-v1 (~85M parameters)
- **Training**: Full finetuning (not LoRA)
- **Hardware**: NVIDIA RTX 3050 6GB
- **Optimization**: Gradient checkpointing + FP16 mixed precision
- **Dataset**: ChEMBL, DrugBank, FDA drugs (approved + withdrawn)

## Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.1f}% |
| **F1 Score** | {f1:.1f}% |
| **ROC-AUC** | {roc_auc:.1f}% |
| **PR-AUC** | {pr_auc:.1f}% |

### Comparison with Pretrained

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Pretrained (no finetuning) | 44.2% | 42.9% |
| **Finetuned (this model)** | **{accuracy:.1f}%** | **{roc_auc:.1f}%** |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/drug-discovery-chemberta")
model = AutoModelForSequenceClassification.from_pretrained("YOUR_USERNAME/drug-discovery-chemberta")

# Predict for a molecule (Aspirin)
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = "Approved" if probs[0][1] > 0.5 else "Failed"
    confidence = probs[0][1].item() if prediction == "Approved" else probs[0][0].item()

print(f"Prediction: {{prediction}} (confidence: {{confidence:.2%}})")
```

## Labels

- **0**: Failed/Withdrawn drug
- **1**: Approved drug

## Intended Use

This model is intended for:
- Drug discovery research
- Virtual screening of candidate molecules
- Prioritizing compounds for further development

## Limitations

- Trained on small molecule drugs only
- Predictions are probabilistic and should not replace clinical trials
- Performance may vary for novel chemical scaffolds

## Citation

If you use this model, please cite:

```bibtex
@misc{{drug-discovery-chemberta-2024,
  author = {{Your Name}},
  title = {{Drug Discovery ChemBERTa: Finetuned for Drug Success Prediction}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/YOUR_USERNAME/drug-discovery-chemberta}}
}}
```

## License

MIT License
'''
    
    output_path = Path(output_dir) / "README.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print(f"Model card created: {output_path}")


def upload_to_huggingface(
    model_dir: str,
    repo_name: str,
    token: str = None,
    private: bool = False
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_dir: Directory containing model files
        repo_name: Repository name (e.g., "username/model-name")
        token: Hugging Face API token
        private: Whether to make the repo private
    """
    api = HfApi()
    
    # Create repo if doesn't exist
    print(f"Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload all files
    print(f"Uploading files from {model_dir}...")
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        token=token
    )
    
    print(f"\n✅ Model uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/{repo_name}")


def main(args):
    """Main upload function."""
    print("="*60)
    print("Upload Drug Discovery Model to Hugging Face")
    print("="*60)
    
    # Check for trained model
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Error: Trained model not found at {checkpoint_path}")
        print("Please run 'python train.py' first.")
        return
    
    # Prepare model files
    upload_dir = Path("hf_upload")
    print("\n[1/3] Preparing model files...")
    prepare_model_for_upload(str(checkpoint_path), str(upload_dir))
    
    # Create model card
    print("\n[2/3] Creating model card...")
    create_model_card(str(upload_dir))
    
    # Upload to Hugging Face
    if args.upload:
        print("\n[3/3] Uploading to Hugging Face...")
        
        if not args.token:
            print("\nTo upload, you need a Hugging Face token.")
            print("1. Go to https://huggingface.co/settings/tokens")
            print("2. Create a new token with 'write' access")
            print("3. Run: python upload_to_hf.py --upload --token YOUR_TOKEN --repo YOUR_USERNAME/drug-discovery-chemberta")
            return
        
        upload_to_huggingface(
            model_dir=str(upload_dir),
            repo_name=args.repo,
            token=args.token,
            private=args.private
        )
    else:
        print("\n[3/3] Skipping upload (use --upload to push to Hub)")
        print(f"\nModel files prepared in: {upload_dir.absolute()}")
        print("\nTo upload, run:")
        print(f"  python upload_to_hf.py --upload --token YOUR_TOKEN --repo YOUR_USERNAME/drug-discovery-chemberta")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload model to Hugging Face Hub"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually upload to Hugging Face (otherwise just prepares files)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (get from https://huggingface.co/settings/tokens)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="drug-discovery-chemberta",
        help="Repository name (e.g., username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    main(args)
