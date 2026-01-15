"""
Upload Drug Discovery Text Generation Model to Hugging Face Hub

Publishes the finetuned Qwen2.5-14B model for drug discovery.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from config import MODEL_CONFIG, CHECKPOINT_DIR, RESULTS_DIR

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, login
    HF_AVAILABLE = True
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import HfApi, create_repo, upload_folder, login
    HF_AVAILABLE = True


def find_latest_checkpoint():
    """Find the latest training checkpoint."""
    checkpoints = list(CHECKPOINT_DIR.glob("run_*/final"))
    if checkpoints:
        return sorted(checkpoints)[-1]
    return None


def create_model_card(output_dir: Path, metrics: dict = None):
    """Create README.md model card for Hugging Face."""
    
    # Default metrics
    if metrics is None:
        metrics_path = RESULTS_DIR / "generation_eval_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
    
    keyword_coverage = metrics.get('avg_keyword_coverage', 0) * 100
    success_rate = metrics.get('success_rate', 0) * 100
    avg_length = metrics.get('avg_response_length', 0)
    
    model_card = f'''---
language: en
license: mit
tags:
  - drug-discovery
  - qwen2
  - text-generation
  - pharmaceutical
  - molecular-analysis
  - smiles
  - chemistry
  - healthcare
datasets:
  - chembl
  - fda
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-14B-Instruct
---

# Drug Discovery AI - Qwen2.5-14B Finetuned

An AI-powered drug discovery assistant that provides detailed analysis, failure explanations, and improvement suggestions for pharmaceutical research.

## Model Description

This model is a finetuned version of [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) for drug discovery assistance.

### Capabilities

- **Drug Analysis**: Predict approval likelihood with detailed explanations
- **Failure Analysis**: Understand why drugs fail with mechanistic insights
- **Drug Comparison**: Compare safety profiles of two candidates
- **Improvement Suggestions**: Get structural modification recommendations
- **Property Analysis**: Analyze molecular properties and drug-likeness

### Training Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-14B-Instruct (14.7B parameters) |
| **Training Type** | Full fine-tuning (SFTTrainer) |
| **Hardware** | AMD MI300X 192GB |
| **Precision** | BFloat16 |
| **Dataset** | ChEMBL, FDA, clinical trial failures |
| **Training Samples** | ~11,000 |

## Performance

| Metric | Score |
|--------|-------|
| **Keyword Coverage** | {keyword_coverage:.1f}% |
| **Success Rate** | {success_rate:.1f}% |
| **Avg Response Length** | {avg_length:.0f} chars |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_name = "YOUR_USERNAME/drug-discovery-qwen-14b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Analyze a drug
prompt = """<|im_start|>user
Analyze this drug candidate and predict its approval likelihood:
SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
Drug Name: Aspirin
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Example Output

**Input:**
```
Analyze this drug candidate: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)
```

**Output:**
```markdown
## Drug Analysis Report

### Prediction: ✅ APPROVED
**Confidence Level:** High

### Molecular Properties Analysis
| Property | Value | Assessment |
|----------|-------|------------|
| Molecular Weight | 180.2 Da | ✓ Good (<500) |
| LogP | 1.19 | ✓ Good (<5) |
| Rule of 5 Violations | 0 | ✓ Compliant |

### Drug-Likeness Assessment
The compound shows excellent drug-like properties with full Lipinski compliance...
```

## Intended Use

- Pharmaceutical research and development
- Drug candidate screening
- Understanding drug failures and safety issues
- Educational purposes in medicinal chemistry

## Limitations

- Predictions are probabilistic and should not replace clinical trials
- Trained primarily on small molecule drugs
- May have reduced accuracy for novel chemical scaffolds
- Should be used as a research tool, not for clinical decisions

## Citation

```bibtex
@misc{{drug-discovery-qwen-2024,
  author = {{Prashanth Kumar}},
  title = {{Drug Discovery AI: Qwen2.5-14B Finetuned for Pharmaceutical Research}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/YOUR_USERNAME/drug-discovery-qwen-14b}}
}}
```

## License

MIT License

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [ChEMBL](https://www.ebi.ac.uk/chembl/) for bioactivity data
- [FDA](https://open.fda.gov/) for drug approval data
- AMD for MI300X GPU credits
'''
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print(f"Model card created: {readme_path}")


def upload_to_huggingface(
    model_dir: str,
    repo_name: str,
    token: str = None,
    private: bool = False
):
    """Upload model to Hugging Face Hub."""
    
    # Login if token provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    # Create repo if doesn't exist
    print(f"Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload all files
    print(f"Uploading files from {model_dir}...")
    print("This may take a while for a 14B model...")
    
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
    )
    
    print(f"\n✅ Model uploaded successfully!")
    print(f"🔗 View at: https://huggingface.co/{repo_name}")


def main(args):
    """Main upload function."""
    print("="*60)
    print("Upload Drug Discovery Model to Hugging Face")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find model checkpoint
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = find_latest_checkpoint()
    
    if model_path is None or not model_path.exists():
        print(f"Error: Model not found!")
        print("Please specify --model_path or ensure checkpoints exist.")
        print("\nUsage:")
        print("  python upload_to_hf.py --model_path checkpoints/run_XXXXXX/final --repo YOUR_USERNAME/drug-discovery-qwen-14b")
        return
    
    print(f"Model path: {model_path}")
    
    # Create model card in the model directory
    print("\n[1/2] Creating model card...")
    create_model_card(model_path)
    
    # Upload to Hugging Face
    if args.upload:
        print("\n[2/2] Uploading to Hugging Face...")
        
        if not args.token:
            # Check for HF_TOKEN environment variable
            token = os.environ.get("HF_TOKEN")
            if not token:
                print("\n⚠️ No token provided!")
                print("Set HF_TOKEN environment variable or use --token flag:")
                print("  export HF_TOKEN=your_token")
                print("  python upload_to_hf.py --upload --repo YOUR_USERNAME/drug-discovery-qwen-14b")
                return
        else:
            token = args.token
        
        upload_to_huggingface(
            model_dir=str(model_path),
            repo_name=args.repo,
            token=token,
            private=args.private
        )
    else:
        print("\n[2/2] Skipping upload (use --upload to push to Hub)")
        print(f"\nModel ready at: {model_path}")
        print("\nTo upload, run:")
        print(f"  python upload_to_hf.py --upload --model_path {model_path} --repo YOUR_USERNAME/drug-discovery-qwen-14b")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload drug discovery model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint directory"
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
        help="Hugging Face API token (or set HF_TOKEN env variable)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="drug-discovery-qwen-14b",
        help="Repository name (e.g., username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    main(args)
