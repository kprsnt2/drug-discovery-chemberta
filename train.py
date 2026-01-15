"""
Training Script for Drug Discovery Text Generation Model

Full fine-tuning of Qwen2.5-14B for drug discovery assistance.
Optimized for AMD MI300X 192GB VRAM.

Usage:
    python train.py --epochs 3
    python train.py --test_run  # Quick test with minimal data
"""

import os
import sys
import argparse
import time
import json
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG,
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGGING_CONFIG
)
from src.model import DrugDiscoveryLLM, get_tokenizer, estimate_memory_usage

try:
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
    )
    from datasets import Dataset, load_dataset
except ImportError:
    print("Installing required packages...")
    os.system("pip install transformers datasets accelerate")
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
    )
    from datasets import Dataset, load_dataset

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    print("Installing trl...")
    os.system("pip install trl>=0.7.0")
    from trl import SFTTrainer, SFTConfig


def setup_device():
    """Set up training device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Enable TF32 for supported GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction
        # torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available. Training on CPU will be extremely slow!")
    
    return device


def load_instruction_dataset(
    train_path: Path = None,
    val_path: Path = None,
    tokenizer = None,
    max_seq_length: int = 2048,
    test_mode: bool = False,
):
    """
    Load instruction-tuning dataset.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: Tokenizer for encoding
        max_seq_length: Maximum sequence length
        test_mode: If True, use small subset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    instruct_dir = PROCESSED_DATA_DIR / "instruct"
    
    train_path = train_path or instruct_dir / "train_instruct.jsonl"
    val_path = val_path or instruct_dir / "val_instruct.jsonl"
    
    # Check if instruction dataset exists
    if not train_path.exists():
        print(f"Instruction dataset not found at {train_path}")
        print("Please run: python scripts/prepare_instruct_dataset.py")
        
        # Try to create it
        print("\nAttempting to create instruction dataset...")
        from scripts.prepare_instruct_dataset import create_instruction_tuning_dataset
        create_instruction_tuning_dataset(instruct_dir)
    
    # Load datasets
    print(f"Loading training data from {train_path}...")
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    print(f"Loading validation data from {val_path}...")
    val_data = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Limit data for test mode
    if test_mode:
        train_data = train_data[:100]
        val_data = val_data[:20]
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def create_training_arguments(
    output_dir: Path,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    test_mode: bool = False,
) -> TrainingArguments:
    """
    Create training arguments for the trainer.
    
    Args:
        output_dir: Directory to save outputs
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        test_mode: If True, use minimal settings
        
    Returns:
        TrainingArguments instance
    """
    epochs = epochs or TRAINING_CONFIG['epochs']
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    
    if test_mode:
        epochs = 1
        batch_size = 2
    
    return SFTConfig(
        output_dir=str(output_dir),
        
        # SFT-specific settings
        max_seq_length=TRAINING_CONFIG['max_seq_length'],
        packing=TRAINING_CONFIG.get('packing', False),
        
        # Training settings
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=TRAINING_CONFIG['weight_decay'],
        adam_beta1=TRAINING_CONFIG['adam_beta1'],
        adam_beta2=TRAINING_CONFIG['adam_beta2'],
        adam_epsilon=TRAINING_CONFIG['adam_epsilon'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        
        # Scheduler
        warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
        lr_scheduler_type=TRAINING_CONFIG['lr_scheduler_type'],
        
        # Precision
        bf16=TRAINING_CONFIG['bf16'],
        fp16=TRAINING_CONFIG['fp16'],
        
        # Memory optimization
        gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing'],
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=TRAINING_CONFIG['logging_steps'],
        report_to="none",  # Set to "wandb" for experiment tracking
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG['eval_steps'],
        
        # Saving
        save_strategy="steps",
        save_steps=TRAINING_CONFIG['save_steps'],
        save_total_limit=TRAINING_CONFIG['save_total_limit'],
        load_best_model_at_end=TRAINING_CONFIG['load_best_model_at_end'],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Data loading
        dataloader_num_workers=TRAINING_CONFIG['dataloader_num_workers'],
        dataloader_pin_memory=TRAINING_CONFIG['pin_memory'],
        
        # Other
        remove_unused_columns=False,
        disable_tqdm=False,
    )


def main(args):
    """Main training function."""
    print("="*60)
    print("Drug Discovery Text Generation Model Training")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = setup_device()
    
    # Memory estimation
    print("\n[1/5] Memory estimation...")
    memory = estimate_memory_usage(14, args.batch_size, 2048)
    print(f"  Estimated memory: {memory['total_gb']:.1f} GB")
    print(f"  Available: 192 GB")
    print(f"  Headroom: {192 - memory['total_gb']:.1f} GB")
    
    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"  Vocabulary size: {len(tokenizer)}")
    
    # Load dataset
    print("\n[3/5] Loading instruction-tuning dataset...")
    train_dataset, val_dataset = load_instruction_dataset(
        tokenizer=tokenizer,
        max_seq_length=TRAINING_CONFIG['max_seq_length'],
        test_mode=args.test_run,
    )
    
    # Load model
    print("\n[4/5] Loading model...")
    model = DrugDiscoveryLLM(
        model_name=args.model_name or MODEL_CONFIG['model_name'],
        torch_dtype=MODEL_CONFIG['torch_dtype'],
        use_flash_attention=MODEL_CONFIG['use_flash_attention'],
    )
    
    # Prepare for training
    model.prepare_for_training(use_lora=args.use_lora)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = CHECKPOINT_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = create_training_arguments(
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_mode=args.test_run,
    )
    
    # Create SFT Trainer
    print("\n[5/5] Setting up trainer...")
    
    # Response template for completion-only training
    response_template = "<|im_start|>assistant\n"
    
    trainer = SFTTrainer(
        model=model.model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
                early_stopping_threshold=TRAINING_CONFIG['early_stopping_threshold'],
            )
        ],
    )
    
    # Print training info
    print("\n" + "-"*40)
    print("Training Configuration")
    print("-"*40)
    print(f"  Model: {MODEL_CONFIG['model_name']}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Effective batch: {args.batch_size * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {TRAINING_CONFIG['max_seq_length']}")
    print(f"  LoRA: {args.use_lora}")
    print(f"  Output: {output_dir}")
    
    # Start training
    print("\n" + "="*40)
    print("Starting training...")
    print("="*40 + "\n")
    
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        
        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", training_time),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
        
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"Final train loss: {train_result.training_loss:.4f}")
        print(f"Model saved to: {output_dir / 'final'}")
        
        # Generate sample output
        print("\n" + "-"*40)
        print("Sample Generation Test")
        print("-"*40)
        
        test_prompt = "Analyze this drug candidate and predict its approval likelihood:\nSMILES: CC(=O)OC1=CC=CC=C1C(=O)O\nDrug Name: Aspirin"
        
        model.model.eval()
        output = model.generate(test_prompt, max_new_tokens=512)
        print(f"\nPrompt: {test_prompt[:100]}...")
        print(f"\nGenerated response:\n{output.generated_text[:500]}...")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Test the model: python app.py")
    print(f"2. Push to HuggingFace: python upload_to_hf.py --model_path {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train drug discovery text generation model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name (default from config)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=TRAINING_CONFIG['epochs'],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_CONFIG['batch_size'],
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=TRAINING_CONFIG['learning_rate'],
        help="Learning rate"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training (default: full fine-tuning)"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Quick test run with minimal training"
    )
    
    args = parser.parse_args()
    
    # Override for test run
    if args.test_run:
        args.epochs = 1
        print("TEST RUN MODE - Using minimal configuration")
    
    main(args)
