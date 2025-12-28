# Cloud Training Guide for Drug Discovery with GPT-OSS-120B

One-click deployment with **ROCm 7 + GPT-OSS-120B** on AMD MI300X.

---

## Quick Start (Pre-configured Cloud with ROCm 7 + GPT-OSS)

If your cloud instance has ROCm 7 and GPT-OSS pre-installed, follow these simplified steps:

### Step 1: Push Code to GitHub (Local Machine)

```powershell
cd "c:\Users\Prashanth Kumar\Desktop\DrugDisc"
git add .
git commit -m "Add cloud training for AMD MI300X with GPT-OSS-120B"
git push origin main
```

### Step 2: Clone on Cloud GPU

```bash
git clone https://github.com/kprsnt2/drug-discovery-chemberta.git
cd drug-discovery-chemberta
```

### Step 3: Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment (required for Python 3.12+)
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install transformers datasets accelerate deepspeed peft
pip install pandas scikit-learn rdkit matplotlib seaborn tqdm wandb
pip install chembl-webresource-client requests huggingface_hub
```

### Step 4: Download Drug Datasets

```bash
python scripts/download_all.py
```

### Step 5: Train GPT-OSS-120B

```bash
# Uses pre-installed GPT-OSS-120B model
python cloud/train_cloud.py --gpu mi300x --model gpt-oss-120b
```

### Step 6: Evaluate & Benchmark

```bash
python evaluate.py
python benchmark.py --generate_report
```

### Step 7: Upload Finetuned Model

```bash
huggingface-cli login
python upload_to_hf.py --upload --repo YOUR_USERNAME/gpt-oss-120b-drug-discovery
```

---

## Full Setup (Manual Installation)

<details>
<summary>Click to expand manual setup steps</summary>

### Install ROCm PyTorch (if not pre-installed)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r cloud/requirements_cloud.txt
```

### Verify GPU

```bash
python -c "import torch; print(f'ROCm: {torch.version.hip}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB')"
```

### Download GPT-OSS-120B (if not pre-installed)

```bash
huggingface-cli download openai/gpt-oss-120b --local-dir models/gpt-oss-120b
```

</details>

---

## Available Models

| Key | Model | Params | Min VRAM |
|-----|-------|--------|----------|
| `chemberta` | ChemBERTa | 85M | 4GB |
| `llama-3.1-8b` | Llama 3.1 8B | 8B | 24GB |
| `llama-3.1-70b` | Llama 3.1 70B | 70B | 140GB |
| **`gpt-oss-120b`** | **GPT-OSS 120B** ⭐ | **120B** | **180GB** |

---

## Training Commands

```bash
# Default (auto-selects GPT-OSS-120B for MI300X)
python cloud/train_cloud.py --gpu mi300x

# With WandB logging
python cloud/train_cloud.py --gpu mi300x --wandb

# Quick test run
python cloud/train_cloud.py --gpu mi300x --max_samples 100

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=4 cloud/train_cloud.py --gpu mi300x --deepspeed
```

---

## Estimated Training Time

| Model | Time (full dataset) |
|-------|-------------------|
| GPT-OSS-120B on MI300X | ~12 hours |
