---
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
---

# Drug Discovery Model - Qwen 2.5 14B Finetuned

This model is a fine-tuned version of **Qwen 2.5 14B** for **drug discovery and molecular property prediction**. It predicts whether a drug compound (represented as a SMILES string) is likely to be approved or fail in clinical trials.

## Model Description

- **Base Model:** Qwen/Qwen2.5-14B
- **Task:** Binary classification (Drug Approval Prediction)
- **Training Hardware:** AMD MI300X 192GB GPU
- **Training Framework:** HuggingFace Transformers with 4-bit quantization

## Intended Uses

This model is designed for:
- **Drug discovery research** - Predicting drug candidate success
- **Virtual screening** - Filtering compounds before synthesis
- **Lead optimization** - Evaluating molecular modifications
- **Educational purposes** - Learning about AI in drug discovery

## Training Data

The model was trained on a curated dataset of drug molecules with known clinical trial outcomes:
- **Format:** SMILES molecular representations
- **Labels:** Binary (0 = Failed, 1 = Approved)
- **Source:** Public drug databases and clinical trial records

## Training Procedure

### Training Hyperparameters

- **Epochs:** 3
- **Batch Size:** 1 (with gradient accumulation)
- **Learning Rate:** 1e-5
- **Optimizer:** AdamW
- **Precision:** bfloat16 with 4-bit quantization
- **Gradient Checkpointing:** Enabled

### Hardware

- **GPU:** AMD Instinct MI300X (192GB HBM3)
- **Backend:** ROCm
- **Training Time:** ~2-3 hours

## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| F1 Score | TBD |
| ROC-AUC | TBD |
| PR-AUC | TBD |

*Note: Update these results after running evaluation*

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

# Tokenize
inputs = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()

print(f"Prediction: {'Approved' if prediction == 1 else 'Failed'}")
print(f"Confidence: {probs[0][prediction].item():.2%}")
```

## Limitations

- Model predictions should not replace expert judgment in drug development
- Performance may vary on molecules very different from training data
- SMILES representation may not capture all relevant molecular properties
- Model was trained on historical data and may not reflect current approval criteria

## Citation

If you use this model, please cite:

```bibtex
@misc{drug-discovery-qwen-14b,
  author = {Prashanth Kumar},
  title = {Drug Discovery Model - Qwen 2.5 14B Finetuned},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/kprsnt2/drug-discovery-qwen-14b}
}
```

## Model Card Contact

For questions or feedback, please open an issue on the model repository.
