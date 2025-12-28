"""
Configuration settings for Drug Discovery Model Finetuning
Optimized for RTX 3050 6GB VRAM
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
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    # ChemBERTa-77M-MLM - Small enough for full finetuning on 6GB GPU
    "model_name": "seyonec/ChemBERTa-zinc-base-v1",
    "num_labels": 2,  # Binary: approved/failed
    "max_length": 128,  # Token limit for SMILES
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
}

# ============================================================================
# TRAINING CONFIGURATION - Optimized for 6GB VRAM
# ============================================================================
TRAINING_CONFIG = {
    # Batch settings
    "batch_size": 8,  # Small batch for memory
    "gradient_accumulation_steps": 4,  # Effective batch = 32
    "max_grad_norm": 1.0,
    
    # Training duration
    "epochs": 10,
    "warmup_ratio": 0.1,
    
    # Optimizer
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    
    # Memory optimization - CRITICAL for 6GB VRAM
    "gradient_checkpointing": True,  # Saves ~40% memory
    "fp16": True,  # Mixed precision - saves ~50% memory
    "dataloader_num_workers": 2,
    "pin_memory": True,
    
    # Checkpointing
    "save_steps": 500,
    "eval_steps": 100,
    "logging_steps": 50,
    "save_total_limit": 3,
    
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    # Data splits
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Filtering
    "min_smiles_length": 5,
    "max_smiles_length": 200,
    
    # Labels
    "label_mapping": {
        "approved": 1,
        "failed": 0,
        "withdrawn": 0,
        "experimental": 0,  # Treat as not-yet-approved
    },
    
    # Augmentation
    "smiles_augmentation": True,
    "augmentation_factor": 2,  # Generate 2x augmented samples
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    "metrics": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "classification_threshold": 0.5,
    "generate_confusion_matrix": True,
    "generate_roc_curve": True,
}

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================
BENCHMARK_CONFIG = {
    "compare_with_pretrained": True,
    "random_baseline": True,
    "cross_validation_folds": 5,
    "generate_plots": True,
    "output_format": ["csv", "json", "html"],
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": BASE_DIR / "training.log",
    "tensorboard_dir": BASE_DIR / "runs",
}
