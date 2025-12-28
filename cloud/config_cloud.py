"""
Cloud Training Configuration for Large Models

Supports:
- AMD MI300X (192GB VRAM) with ROCm
- NVIDIA H100/A100 with CUDA
- Models up to 100B+ parameters
- DeepSpeed ZeRO optimization
"""

import os
import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

# ============================================================================
# GPU PROFILES
# ============================================================================

class GPUProfile(Enum):
    """Available GPU profiles."""
    NVIDIA_RTX3050_6GB = "rtx3050"      # Local consumer GPU
    NVIDIA_RTX4090_24GB = "rtx4090"     # Local high-end
    NVIDIA_A100_40GB = "a100_40"        # Cloud
    NVIDIA_A100_80GB = "a100_80"        # Cloud
    NVIDIA_H100_80GB = "h100"           # Cloud
    AMD_MI300X_192GB = "mi300x"         # Cloud (ROCm)


@dataclass
class GPUConfig:
    """Configuration for a specific GPU."""
    name: str
    vram_gb: int
    backend: str  # "cuda" or "rocm"
    max_batch_size: int
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    deepspeed_stage: int  # 0=None, 1, 2, 3
    recommended_model: str
    max_model_params: str  # e.g., "100M", "70B"


# GPU configurations
GPU_CONFIGS: Dict[GPUProfile, GPUConfig] = {
    GPUProfile.NVIDIA_RTX3050_6GB: GPUConfig(
        name="NVIDIA RTX 3050 6GB",
        vram_gb=6,
        backend="cuda",
        max_batch_size=8,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        deepspeed_stage=0,
        recommended_model="seyonec/ChemBERTa-zinc-base-v1",
        max_model_params="100M"
    ),
    GPUProfile.NVIDIA_RTX4090_24GB: GPUConfig(
        name="NVIDIA RTX 4090 24GB",
        vram_gb=24,
        backend="cuda",
        max_batch_size=32,
        gradient_checkpointing=True,
        fp16=True,
        bf16=True,
        deepspeed_stage=2,
        recommended_model="meta-llama/Llama-3.2-3B",
        max_model_params="7B"
    ),
    GPUProfile.NVIDIA_A100_40GB: GPUConfig(
        name="NVIDIA A100 40GB",
        vram_gb=40,
        backend="cuda",
        max_batch_size=64,
        gradient_checkpointing=True,
        fp16=True,
        bf16=True,
        deepspeed_stage=2,
        recommended_model="meta-llama/Llama-3.1-8B",
        max_model_params="13B"
    ),
    GPUProfile.NVIDIA_A100_80GB: GPUConfig(
        name="NVIDIA A100 80GB",
        vram_gb=80,
        backend="cuda",
        max_batch_size=128,
        gradient_checkpointing=True,
        fp16=True,
        bf16=True,
        deepspeed_stage=2,
        recommended_model="meta-llama/Llama-3.1-70B",
        max_model_params="70B"
    ),
    GPUProfile.NVIDIA_H100_80GB: GPUConfig(
        name="NVIDIA H100 80GB",
        vram_gb=80,
        backend="cuda",
        max_batch_size=128,
        gradient_checkpointing=False,
        fp16=True,
        bf16=True,
        deepspeed_stage=2,
        recommended_model="meta-llama/Llama-3.1-70B",
        max_model_params="70B"
    ),
    GPUProfile.AMD_MI300X_192GB: GPUConfig(
        name="AMD MI300X 192GB",
        vram_gb=192,
        backend="rocm",
        max_batch_size=4,  # Small batch - 120B model uses ~188GB
        gradient_checkpointing=True,  # Needed for training 120B
        fp16=True,
        bf16=True,
        deepspeed_stage=2,
        recommended_model="openai/gpt-oss-120b",
        max_model_params="120B"
    ),
}


# ============================================================================
# MODEL PROFILES
# ============================================================================

class ModelSize(Enum):
    """Model size categories."""
    SMALL = "small"      # < 1B params
    MEDIUM = "medium"    # 1B - 10B params
    LARGE = "large"      # 10B - 70B params
    XLARGE = "xlarge"    # 70B+ params


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    hf_name: str
    params: str
    size: ModelSize
    min_vram_gb: int
    max_length: int
    is_causal_lm: bool
    supports_flash_attention: bool


# Available models for drug discovery
MODELS: Dict[str, ModelConfig] = {
    # Small models (works on consumer GPUs)
    "chemberta": ModelConfig(
        name="ChemBERTa",
        hf_name="seyonec/ChemBERTa-zinc-base-v1",
        params="85M",
        size=ModelSize.SMALL,
        min_vram_gb=4,
        max_length=512,
        is_causal_lm=False,
        supports_flash_attention=False
    ),
    "chemberta-mlm": ModelConfig(
        name="ChemBERTa-77M-MLM",
        hf_name="DeepChem/ChemBERTa-77M-MLM",
        params="77M",
        size=ModelSize.SMALL,
        min_vram_gb=4,
        max_length=512,
        is_causal_lm=False,
        supports_flash_attention=False
    ),
    
    # Medium models (requires 24GB+)
    "llama-3.2-3b": ModelConfig(
        name="Llama 3.2 3B",
        hf_name="meta-llama/Llama-3.2-3B",
        params="3B",
        size=ModelSize.MEDIUM,
        min_vram_gb=16,
        max_length=8192,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        hf_name="mistralai/Mistral-7B-v0.3",
        params="7B",
        size=ModelSize.MEDIUM,
        min_vram_gb=24,
        max_length=32768,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        hf_name="meta-llama/Llama-3.1-8B",
        params="8B",
        size=ModelSize.MEDIUM,
        min_vram_gb=24,
        max_length=131072,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    
    # Large models (requires 80GB+)
    "llama-3.1-70b": ModelConfig(
        name="Llama 3.1 70B",
        hf_name="meta-llama/Llama-3.1-70B",
        params="70B",
        size=ModelSize.LARGE,
        min_vram_gb=140,
        max_length=131072,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    "qwen-72b": ModelConfig(
        name="Qwen 2.5 72B",
        hf_name="Qwen/Qwen2.5-72B",
        params="72B",
        size=ModelSize.LARGE,
        min_vram_gb=144,
        max_length=131072,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    
    # XLarge models (requires 192GB+)
    "gpt-oss-120b": ModelConfig(
        name="GPT-OSS 120B",
        hf_name="openai/gpt-oss-120b",
        params="120B",
        size=ModelSize.XLARGE,
        min_vram_gb=180,  # Fits on single MI300X with MXFP4
        max_length=131072,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
    "llama-3.1-405b": ModelConfig(
        name="Llama 3.1 405B (Quantized)",
        hf_name="meta-llama/Llama-3.1-405B",
        params="405B",
        size=ModelSize.XLARGE,
        min_vram_gb=400,  # Needs quantization for single GPU
        max_length=131072,
        is_causal_lm=True,
        supports_flash_attention=True
    ),
}


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TRAINING CONFIGURATION FACTORY
# ============================================================================

def get_training_config(
    gpu_profile: GPUProfile,
    model_key: str = None,
    custom_batch_size: int = None
) -> Dict[str, Any]:
    """
    Generate training configuration based on GPU and model.
    
    Args:
        gpu_profile: GPU profile to use
        model_key: Model key from MODELS dict
        custom_batch_size: Override batch size
    
    Returns:
        Dictionary with training configuration
    """
    gpu_config = GPU_CONFIGS[gpu_profile]
    
    # Auto-select model if not specified
    if model_key is None:
        model_key = "chemberta" if gpu_config.vram_gb < 16 else "llama-3.1-8b"
        if gpu_config.vram_gb >= 140:
            model_key = "llama-3.1-70b"
        if gpu_config.vram_gb >= 180:
            model_key = "gpt-oss-120b"  # Prefer GPT-OSS for MI300X
    
    model_config = MODELS.get(model_key)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_key}")
    
    # Check VRAM compatibility
    if model_config.min_vram_gb > gpu_config.vram_gb:
        print(f"WARNING: {model_config.name} requires {model_config.min_vram_gb}GB, "
              f"but {gpu_config.name} has {gpu_config.vram_gb}GB. "
              "Consider using quantization or DeepSpeed ZeRO-3.")
    
    # Calculate optimal batch size
    if custom_batch_size:
        batch_size = custom_batch_size
    else:
        # Heuristic: more VRAM = larger batch
        vram_ratio = gpu_config.vram_gb / model_config.min_vram_gb
        batch_size = min(int(gpu_config.max_batch_size * vram_ratio), 256)
        batch_size = max(batch_size, 1)
    
    # Calculate gradient accumulation for effective batch size of 32-64
    target_effective_batch = 64 if gpu_config.vram_gb >= 80 else 32
    gradient_accumulation = max(1, target_effective_batch // batch_size)
    
    return {
        # GPU settings
        "gpu": {
            "profile": gpu_profile.value,
            "name": gpu_config.name,
            "vram_gb": gpu_config.vram_gb,
            "backend": gpu_config.backend,
        },
        
        # Model settings
        "model": {
            "name": model_config.name,
            "hf_name": model_config.hf_name,
            "params": model_config.params,
            "max_length": min(model_config.max_length, 4096),  # Cap for training
            "is_causal_lm": model_config.is_causal_lm,
            "num_labels": 2,
        },
        
        # Training settings
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "effective_batch_size": batch_size * gradient_accumulation,
            "epochs": 5 if model_config.size in [ModelSize.LARGE, ModelSize.XLARGE] else 10,
            "learning_rate": 1e-5 if model_config.size in [ModelSize.LARGE, ModelSize.XLARGE] else 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        
        # Memory optimization
        "memory": {
            "gradient_checkpointing": gpu_config.gradient_checkpointing,
            "fp16": gpu_config.fp16 and gpu_config.backend == "cuda",
            "bf16": gpu_config.bf16,
            "flash_attention": model_config.supports_flash_attention and gpu_config.vram_gb >= 16,
        },
        
        # DeepSpeed (for distributed/large models)
        "deepspeed": {
            "enabled": gpu_config.deepspeed_stage > 0,
            "stage": gpu_config.deepspeed_stage,
            "offload_optimizer": gpu_config.vram_gb < model_config.min_vram_gb,
            "offload_param": gpu_config.vram_gb < model_config.min_vram_gb * 0.5,
        },
        
        # Checkpointing
        "checkpointing": {
            "save_steps": 500,
            "eval_steps": 100,
            "save_total_limit": 3,
        },
        
        # Logging
        "logging": {
            "log_steps": 10,
            "use_wandb": True,
            "project_name": "drug-discovery-llm",
        }
    }


def print_config(config: Dict[str, Any]):
    """Pretty print configuration."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    print(f"\n🖥️  GPU: {config['gpu']['name']} ({config['gpu']['vram_gb']}GB)")
    print(f"   Backend: {config['gpu']['backend'].upper()}")
    
    print(f"\n🤖 Model: {config['model']['name']}")
    print(f"   HuggingFace: {config['model']['hf_name']}")
    print(f"   Parameters: {config['model']['params']}")
    
    print(f"\n⚙️  Training:")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"   Effective batch: {config['training']['effective_batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Epochs: {config['training']['epochs']}")
    
    print(f"\n💾 Memory Optimization:")
    print(f"   Gradient checkpointing: {config['memory']['gradient_checkpointing']}")
    print(f"   FP16: {config['memory']['fp16']}")
    print(f"   BF16: {config['memory']['bf16']}")
    print(f"   Flash Attention: {config['memory']['flash_attention']}")
    
    if config['deepspeed']['enabled']:
        print(f"\n🚀 DeepSpeed:")
        print(f"   Stage: ZeRO-{config['deepspeed']['stage']}")
        print(f"   Offload optimizer: {config['deepspeed']['offload_optimizer']}")
        print(f"   Offload params: {config['deepspeed']['offload_param']}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Show training configurations")
    parser.add_argument("--gpu", type=str, default="mi300x",
                       choices=[p.value for p in GPUProfile],
                       help="GPU profile to use")
    parser.add_argument("--model", type=str, default=None,
                       choices=list(MODELS.keys()),
                       help="Model to use")
    parser.add_argument("--list-gpus", action="store_true",
                       help="List available GPU profiles")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    if args.list_gpus:
        print("\n📊 Available GPU Profiles:\n")
        for profile in GPUProfile:
            cfg = GPU_CONFIGS[profile]
            print(f"  {profile.value:15} - {cfg.name:25} ({cfg.vram_gb}GB, {cfg.backend})")
        print()
        sys.exit(0)
    
    if args.list_models:
        print("\n🤖 Available Models:\n")
        for key, model in MODELS.items():
            print(f"  {key:20} - {model.name:25} ({model.params}, min {model.min_vram_gb}GB)")
        print()
        sys.exit(0)
    
    # Get GPU profile
    gpu_profile = None
    for p in GPUProfile:
        if p.value == args.gpu:
            gpu_profile = p
            break
    
    if gpu_profile is None:
        print(f"Unknown GPU profile: {args.gpu}")
        sys.exit(1)
    
    # Generate and print config
    config = get_training_config(gpu_profile, args.model)
    print_config(config)
