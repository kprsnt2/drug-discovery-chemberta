"""
Configuration settings for Drug Discovery Text Generation Model
Optimized for AMD MI300X 192GB VRAM - Full Fine-tuning of Qwen2.5-14B
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION - Qwen2.5-14B for Text Generation
# ============================================================================
MODEL_CONFIG = {
    # Base model - Qwen2.5-14B-Instruct
    "model_name": "Qwen/Qwen2.5-14B-Instruct",
    "model_type": "causal_lm",  # Changed from sequence_classification
    
    # Context settings for drug analysis
    "max_length": 2048,  # Longer context for detailed explanations
    "max_new_tokens": 1024,  # Max tokens to generate
    
    # Precision settings - AMD MI300X compatible
    "torch_dtype": "bfloat16",  # Use bfloat16 for training stability
    "use_flash_attention": False,  # Disabled - not available on ROCm
    
    # Quantization - NOT needed with 192GB VRAM
    "use_4bit": False,
    "use_8bit": False,
    
    # LoRA settings (optional, for faster training)
    "use_lora": False,  # User requested FULL fine-tuning
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
}

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    
    # Stop tokens
    "eos_token_id": None,  # Will be set from tokenizer
    "pad_token_id": None,  # Will be set from tokenizer
}

# ============================================================================
# TRAINING CONFIGURATION - Optimized for AMD MI300X 192GB VRAM
# ============================================================================
TRAINING_CONFIG = {
    # Batch settings - can use large batches with 192GB VRAM
    "batch_size": 4,  # Per-device batch size
    "gradient_accumulation_steps": 8,  # Effective batch = 32
    "max_grad_norm": 1.0,
    
    # Training duration
    "epochs": 3,  # Full fine-tuning typically needs fewer epochs
    "warmup_ratio": 0.1,
    "warmup_steps": None,  # Calculated from warmup_ratio
    
    # Optimizer - AdamW with appropriate settings
    "learning_rate": 2e-5,  # Lower LR for full fine-tuning
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    
    # Scheduler
    "lr_scheduler_type": "cosine",
    
    # Memory optimization for large model
    "gradient_checkpointing": True,  # Save memory even with 192GB
    "fp16": False,  # Use bf16 instead
    "bf16": True,  # Better stability for training
    
    # Data loading
    "dataloader_num_workers": 4,
    "pin_memory": True,
    
    # Sequence settings
    "max_seq_length": 2048,
    "packing": True,  # Pack multiple samples into one sequence
    
    # Checkpointing
    "save_steps": 400,
    "eval_steps": 200,
    "logging_steps": 50,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    
    # Hub settings (for pushing to HuggingFace)
    "push_to_hub": False,
    "hub_model_id": None,
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    # Data splits
    "train_ratio": 0.85,
    "val_ratio": 0.10,
    "test_ratio": 0.05,
    
    # Instruction format
    "instruction_template": """<|im_start|>user
{instruction}{input}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>""",
    
    # Task types for balanced sampling
    "task_types": [
        "drug_analysis",
        "failure_analysis",
        "comparison",
        "improvement_suggestion",
        "property_analysis",
    ],
    
    # Data augmentation
    "augmentation": False,  # Text data doesn't need augmentation
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    "metrics": ["perplexity", "bleu", "rouge"],
    "generation_samples": 10,  # Generate sample outputs during eval
    "save_generation_samples": True,
}

# ============================================================================
# AMD ROCm SPECIFIC SETTINGS
# ============================================================================
AMD_CONFIG = {
    "use_rocm": True,
    "device": "cuda",  # ROCm uses CUDA interface
    "visible_devices": "0",  # Which GPU to use
    
    # Memory settings
    "max_memory": {0: "190GB"},  # Leave some headroom
    
    # Compilation (experimental)
    "use_torch_compile": False,  # Set True for potential speedup
}

# ============================================================================
# INFERENCE CONFIGURATION (for app.py)
# ============================================================================
INFERENCE_CONFIG = {
    "model_path": None,  # Will be set to checkpoint path after training
    "load_in_4bit": False,  # Not needed with 192GB VRAM
    "device_map": "auto",
    
    # Streaming
    "use_streamer": True,  # Stream output for better UX
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": BASE_DIR / "training.log",
    "use_wandb": False,  # Set True for experiment tracking
    "wandb_project": "drug-discovery-llm",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config() -> dict:
    """Get model configuration with any dynamic updates."""
    config = MODEL_CONFIG.copy()
    return config


def get_training_config() -> dict:
    """Get training configuration with any dynamic updates."""
    config = TRAINING_CONFIG.copy()
    return config


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("DRUG DISCOVERY LLM CONFIGURATION")
    print("=" * 60)
    print(f"\nModel: {MODEL_CONFIG['model_name']}")
    print(f"Type: {MODEL_CONFIG['model_type']}")
    print(f"Precision: {MODEL_CONFIG['torch_dtype']}")
    print(f"LoRA: {MODEL_CONFIG['use_lora']}")
    print(f"\nTraining:")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Effective batch: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  BF16: {TRAINING_CONFIG['bf16']}")
    print(f"\nPaths:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    print_config()
