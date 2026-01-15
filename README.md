# Drug Discovery Text Generation Model

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Hugging%20Face-yellow)](https://huggingface.co/kprsnt/drug-discovery-qwen-14b)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered drug discovery assistant using **Qwen2.5-14B** for generating explanatory drug analysis, failure predictions, and improvement suggestions.

## 🎯 Features

- **Drug Analysis**: Predict approval likelihood with detailed explanations
- **Failure Analysis**: Understand why drugs fail with mechanistic insights
- **Drug Comparison**: Compare safety profiles of two candidates
- **Improvement Suggestions**: Get structural modification recommendations
- **Chat Interface**: Open-ended drug discovery discussions

## 🔬 Model

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-14B-Instruct |
| **Training** | Full fine-tuning on AMD MI300X 192GB |
| **Task** | Text Generation (Causal LM) |
| **Context** | 2048 tokens |
| **Data** | ChEMBL, FDA, clinical trial failures |

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/drug-discovery-llm.git
cd drug-discovery-llm
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Download Drug Data
```bash
# Comprehensive drug data with failure reasons
python scripts/download_comprehensive.py

# Or existing data sources
python scripts/download_all.py
```

### 2. Prepare Instruction Dataset
```bash
python scripts/prepare_instruct_dataset.py
```

### 3. Train Model
```bash
# Full fine-tuning on AMD MI300X
python train.py --epochs 3

# Test run with minimal data
python train.py --test_run
```

### 4. Launch App
```bash
python app.py
```

## 🤗 Using the Pretrained Model

```python
from src.model import DrugDiscoveryLLM

# Load model
model = DrugDiscoveryLLM()

# Analyze a drug
result = model.analyze_drug(
    smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
    name="Aspirin"
)
print(result)

# Explain a failure
result = model.explain_failure(
    smiles="CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3",
    name="Rofecoxib (Vioxx)",
    failure_reason="Cardiovascular toxicity"
)
print(result)

# Compare drugs
result = model.compare_drugs(
    drug1_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
    drug1_name="Aspirin",
    drug2_smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    drug2_name="Ibuprofen"
)
print(result)
```

## 📁 Project Structure

```
drug-discovery-llm/
├── config.py                     # Model & training configuration
├── train.py                      # Training script (SFTTrainer)
├── app.py                        # Gradio web interface
├── requirements.txt              # Dependencies
├── scripts/
│   ├── download_comprehensive.py # Enhanced data download
│   ├── download_all.py           # Original data download
│   └── prepare_instruct_dataset.py # Instruction-tuning format
├── src/
│   ├── model.py                  # DrugDiscoveryLLM class
│   ├── dataset.py                # Dataset utilities
│   └── trainer.py                # Training utilities
├── data/
│   ├── raw/                      # Downloaded data
│   └── processed/                # Processed datasets
└── checkpoints/                  # Model checkpoints
```

## 💾 Hardware Requirements

| Configuration | VRAM Required | Notes |
|--------------|---------------|-------|
| Full Fine-tuning | ~50-80 GB | Recommended: MI300X 192GB |
| LoRA Fine-tuning | ~20-30 GB | Use `--use_lora` flag |
| Inference Only | ~28 GB | BF16 precision |

## 📊 Training Data

| Source | Records | Description |
|--------|---------|-------------|
| ChEMBL Compounds | 10,000+ | Approved & trial drugs with properties |
| Drug Mechanisms | 5,000+ | Mechanisms of action |
| Drug Warnings | 1,000+ | Safety warnings and alerts |
| Clinical Failures | 50+ | Failures with detailed reasons |
| Withdrawal Reasons | 20+ | Post-market withdrawals |

## 🎓 Example Output

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
| H-Bond Acceptors | 4 | ✓ Good (≤10) |
| Rule of 5 Violations | 0 | ✓ Compliant |

### Drug-Likeness Assessment
The compound shows excellent drug-like properties...
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Qwen2.5](https://huggingface.co/Qwen) - Base model
- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Bioactivity database
- [OpenFDA](https://open.fda.gov/) - FDA drug data
- AMD - MI300X GPU credits for training
