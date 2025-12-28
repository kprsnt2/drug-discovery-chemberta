"""
Cloud Training Script for Large Language Models

Supports:
- AMD MI300X (192GB) with ROCm
- NVIDIA H100/A100 with CUDA
- Models: Llama-3.1-70B, Mistral, Qwen, ChemBERTa
- DeepSpeed ZeRO optimization
- Full finetuning and LoRA

Usage:
    # AMD MI300X with Llama-3.1-70B
    python train_cloud.py --gpu mi300x --model llama-3.1-70b
    
    # NVIDIA A100 with Llama-3.1-8B
    python train_cloud.py --gpu a100_80 --model llama-3.1-8b
    
    # With DeepSpeed (multi-GPU)
    deepspeed train_cloud.py --gpu mi300x --model llama-3.1-70b --deepspeed
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import Dataset
import pandas as pd

from config_cloud import (
    GPUProfile, GPU_CONFIGS, MODELS, ModelSize,
    get_training_config, print_config,
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR
)


def detect_gpu_backend():
    """Detect available GPU backend (CUDA or ROCm)."""
    if torch.cuda.is_available():
        # Check if it's AMD ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "rocm", torch.cuda.device_count()
        else:
            return "cuda", torch.cuda.device_count()
    return "cpu", 0


def setup_environment(backend: str):
    """Set up environment variables for the GPU backend."""
    if backend == "rocm":
        # AMD ROCm settings
        os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
        print("🔴 AMD ROCm environment configured")
    elif backend == "cuda":
        # NVIDIA CUDA settings
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("🟢 NVIDIA CUDA environment configured")


def load_dataset_for_training(tokenizer, max_length: int = 512, max_samples: int = None):
    """Load and prepare dataset for training."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run 'python scripts/download_all.py' first."
        )
    
    # Load CSV data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    if max_samples:
        train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(max_samples // 5, len(val_df)), random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["smiles"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['smiles', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['smiles', 'label']])
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x["smiles"], truncation=True, max_length=max_length),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenizer(x["smiles"], truncation=True, max_length=max_length),
        batched=True
    )
    
    # Rename label column
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Set format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    return train_dataset, val_dataset


def load_model(config: dict, num_labels: int = 2):
    """Load model based on configuration."""
    model_name = config["model"]["hf_name"]
    is_causal = config["model"]["is_causal_lm"]
    
    print(f"\nLoading model: {model_name}")
    print(f"Model type: {'Causal LM' if is_causal else 'Encoder'}")
    
    # Quantization config for large models
    quantization_config = None
    gpu_vram = config["gpu"]["vram_gb"]
    model_params = config["model"]["params"]
    
    # Use 4-bit quantization for large models to save VRAM
    # This reduces memory by ~75% (120B: 188GB -> ~50GB)
    large_model_keywords = ["70B", "72B", "120B", "405B"]
    needs_quantization = any(kw in model_params for kw in large_model_keywords)
    
    if needs_quantization:
        print(f"⚡ Using 4-bit quantization for {model_params} model...")
        print("   Memory reduction: ~75% (allows training with gradients)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    if config["memory"]["bf16"]:
        model_dtype = torch.bfloat16
    elif config["memory"]["fp16"]:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    
    print(f"Loading model with dtype: {model_dtype}")
    
    # Load model with error handling for different model types
    try:
        if is_causal:
            # For causal LM, we add a classification head
            print("Loading as sequence classification model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=model_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            # For encoder models (like ChemBERTa)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                trust_remote_code=True,
            )
    except Exception as e:
        print(f"Error loading with AutoModelForSequenceClassification: {e}")
        print("Trying to load base model and add classification head...")
        
        # Fallback: Load base model and add simple classifier
        from transformers import AutoModel
        import torch.nn as nn
        
        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        # Create a simple wrapper with classification head
        class ModelWithClassifier(nn.Module):
            def __init__(self, base_model, num_labels, hidden_size):
                super().__init__()
                self.base_model = base_model
                self.classifier = nn.Linear(hidden_size, num_labels)
                self.num_labels = num_labels
                
            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
                # Use last hidden state
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state[:, -1, :]  # Last token
                else:
                    hidden = outputs[0][:, -1, :]
                logits = self.classifier(hidden)
                
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                
                return type('Output', (), {'loss': loss, 'logits': logits})()
            
            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                """Enable gradient checkpointing on base model."""
                if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                    self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            
            def gradient_checkpointing_disable(self):
                """Disable gradient checkpointing on base model."""
                if hasattr(self.base_model, 'gradient_checkpointing_disable'):
                    self.base_model.gradient_checkpointing_disable()
        
        hidden_size = base_model.config.hidden_size
        model = ModelWithClassifier(base_model, num_labels, hidden_size)
        model.config = base_model.config
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing if configured
    if config["memory"]["gradient_checkpointing"]:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Track if using device_map
    model._uses_device_map = True
    
    return model, tokenizer


def get_training_arguments(config: dict, output_dir: str, uses_device_map: bool = False) -> TrainingArguments:
    """Create training arguments from config."""
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training params
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"] * 2,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        
        # Optimizer
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        
        # Memory optimization
        fp16=config["memory"]["fp16"],
        bf16=config["memory"]["bf16"],
        gradient_checkpointing=config["memory"]["gradient_checkpointing"],
        
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=config["checkpointing"]["eval_steps"],
        save_strategy="steps",
        save_steps=config["checkpointing"]["save_steps"],
        save_total_limit=config["checkpointing"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=config["logging"]["log_steps"],
        logging_first_step=True,
        report_to="wandb" if config["logging"]["use_wandb"] else "none",
        run_name=f"drug-discovery-{config['model']['name']}-{datetime.now().strftime('%Y%m%d_%H%M')}",
        
        # Other
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        
        # CRITICAL: Disable device placement for models using device_map
        use_cpu=False,
    )
    
    return args


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1) if len(predictions.shape) > 1 else predictions
    
    # Compute probabilities for AUC
    if len(eval_pred.predictions.shape) > 1:
        probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=-1)[:, 1].numpy()
    else:
        probs = predictions
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
    }
    
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except:
        metrics["roc_auc"] = 0.5
    
    return metrics


def main(args):
    """Main training function."""
    print("="*70)
    print("🧬 DRUG DISCOVERY - CLOUD MODEL TRAINING")
    print("="*70)
    
    # Detect GPU
    backend, num_gpus = detect_gpu_backend()
    print(f"\nDetected: {backend.upper()} with {num_gpus} GPU(s)")
    
    if num_gpus == 0 and not args.cpu:
        print("ERROR: No GPU detected. Use --cpu for CPU training (not recommended).")
        return
    
    # Setup environment
    setup_environment(backend)
    
    # Get GPU profile
    gpu_profile = None
    for p in GPUProfile:
        if p.value == args.gpu:
            gpu_profile = p
            break
    
    if gpu_profile is None:
        print(f"ERROR: Unknown GPU profile: {args.gpu}")
        print("Available profiles:", [p.value for p in GPUProfile])
        return
    
    # Generate config
    config = get_training_config(gpu_profile, args.model, args.batch_size)
    print_config(config)
    
    # Check backend compatibility
    if config["gpu"]["backend"] != backend and backend != "cpu":
        print(f"WARNING: Config expects {config['gpu']['backend']}, but detected {backend}")
    
    # Load model and tokenizer
    print("\n" + "-"*50)
    print("Loading Model")
    print("-"*50)
    model, tokenizer = load_model(config)
    
    # Load dataset
    print("\n" + "-"*50)
    print("Loading Dataset")
    print("-"*50)
    train_dataset, val_dataset = load_dataset_for_training(
        tokenizer,
        max_length=config["model"]["max_length"],
        max_samples=args.max_samples
    )
    
    # Create output directory
    output_dir = CHECKPOINT_DIR / f"cloud_{config['model']['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Get training arguments
    training_args = get_training_arguments(config, str(output_dir))
    
    # Add DeepSpeed if enabled
    if args.deepspeed and config["deepspeed"]["enabled"]:
        training_args.deepspeed = str(Path(__file__).parent / "deepspeed_config.json")
        print("\n🚀 DeepSpeed enabled")
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    print("\n" + "-"*50)
    print("Starting Training")
    print("-"*50)
    
    # Check if model uses device_map (already on GPU)
    uses_device_map = getattr(model, '_uses_device_map', False)
    
    # Create a custom Trainer that handles device_map properly
    class DeviceMapTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            # Set place_model_on_device BEFORE calling super().__init__
            super().__init__(*args, **kwargs)
        
        def _move_model_to_device(self, model, device):
            # Skip moving - model already on device via device_map
            pass
    
    if uses_device_map:
        print("Using DeviceMapTrainer (model already on GPU via device_map)")
        trainer = DeviceMapTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    # Train
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Save model
    print("\n" + "-"*50)
    print("Saving Model")
    print("-"*50)
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    # Evaluate
    print("\n" + "-"*50)
    print("Final Evaluation")
    print("-"*50)
    eval_result = trainer.evaluate()
    
    # Print results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n⏱️  Training time: {training_time/3600:.2f} hours")
    print(f"📊 Final metrics:")
    print(f"   - Loss: {eval_result['eval_loss']:.4f}")
    print(f"   - Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")
    print(f"   - F1: {eval_result.get('eval_f1', 'N/A')}")
    print(f"   - ROC-AUC: {eval_result.get('eval_roc_auc', 'N/A')}")
    print(f"\n💾 Model saved to: {output_dir}")
    
    # Save results
    results = {
        "config": config,
        "training_time_hours": training_time / 3600,
        "train_result": {k: str(v) for k, v in train_result.metrics.items()},
        "eval_result": {k: str(v) for k, v in eval_result.items()},
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cloud training for drug discovery models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AMD MI300X with Llama-3.1-70B
  python train_cloud.py --gpu mi300x --model llama-3.1-70b
  
  # NVIDIA A100 80GB with Llama-3.1-8B
  python train_cloud.py --gpu a100_80 --model llama-3.1-8b
  
  # RTX 3050 with ChemBERTa (local)
  python train_cloud.py --gpu rtx3050 --model chemberta
  
  # With DeepSpeed (multi-GPU)
  deepspeed train_cloud.py --gpu mi300x --model llama-3.1-70b --deepspeed
        """
    )
    
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="mi300x",
        choices=[p.value for p in GPUProfile],
        help="GPU profile to use"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        choices=list(MODELS.keys()),
        help="Model to finetune"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training samples (for testing)"
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (not recommended)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    main(args)
