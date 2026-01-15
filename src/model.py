"""
Drug Discovery LLM Model

Causal Language Model wrapper for drug discovery text generation.
Supports Qwen2.5-14B with full fine-tuning on AMD MI300X.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, GENERATION_CONFIG, AMD_CONFIG

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        BitsAndBytesConfig,
        PreTrainedModel,
        GenerationConfig,
    )
except ImportError:
    print("Installing transformers...")
    os.system("pip install transformers>=4.36.0")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        BitsAndBytesConfig,
        PreTrainedModel,
        GenerationConfig,
    )

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    print("Installing peft...")
    os.system("pip install peft>=0.7.0")
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel


@dataclass
class DrugDiscoveryOutput:
    """Output container for drug discovery model."""
    generated_text: str
    input_text: str
    generation_tokens: int
    model_name: str


class DrugDiscoveryLLM:
    """
    Text Generation Model for Drug Discovery Assistance.
    
    This model provides:
    - Drug approval prediction with explanations
    - Drug failure analysis
    - Safety improvement suggestions
    - Molecular property analysis
    - Drug comparison
    
    Attributes:
        model_name: HuggingFace model name or path
        model: The loaded transformer model
        tokenizer: The tokenizer for the model
        device: Training/inference device
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention: bool = True,
        torch_dtype: str = "bfloat16",
    ):
        """
        Initialize the Drug Discovery LLM.
        
        Args:
            model_name: HuggingFace model name/path (default from config)
            device: Device to use (default: auto-detect)
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            use_flash_attention: Whether to use Flash Attention 2
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        print(f"Initializing DrugDiscoveryLLM")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.torch_dtype}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = self._load_tokenizer()
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            use_flash_attention=use_flash_attention,
        )
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=GENERATION_CONFIG.get("max_new_tokens", 1024),
            temperature=GENERATION_CONFIG.get("temperature", 0.7),
            top_p=GENERATION_CONFIG.get("top_p", 0.9),
            top_k=GENERATION_CONFIG.get("top_k", 50),
            do_sample=GENERATION_CONFIG.get("do_sample", True),
            repetition_penalty=GENERATION_CONFIG.get("repetition_penalty", 1.1),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        self._print_model_info()
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",  # Important for generation
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer
    
    def _load_model(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention: bool = True,
    ) -> PreTrainedModel:
        """Load the model with specified configuration."""
        
        # Quantization config (if needed)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
        }
        
        # Add quantization config if applicable
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add flash attention if supported
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        return model
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU memory used: {memory_gb:.2f} GB")
    
    def prepare_for_training(self, use_lora: bool = False, lora_config: dict = None):
        """
        Prepare model for training with optional LoRA.
        
        Args:
            use_lora: Whether to apply LoRA for parameter-efficient training
            lora_config: LoRA configuration overrides
        """
        if use_lora:
            print("Applying LoRA for parameter-efficient training...")
            
            config = lora_config or {}
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.get("r", MODEL_CONFIG.get("lora_r", 64)),
                lora_alpha=config.get("alpha", MODEL_CONFIG.get("lora_alpha", 128)),
                lora_dropout=config.get("dropout", MODEL_CONFIG.get("lora_dropout", 0.05)),
                target_modules=config.get("target_modules", MODEL_CONFIG.get("lora_target_modules")),
                bias="none",
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            print("Preparing for full fine-tuning...")
            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            
            # Ensure all parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True
        
        return self
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        stream: bool = False,
        **kwargs
    ) -> DrugDiscoveryOutput:
        """
        Generate response for a drug discovery prompt.
        
        Args:
            prompt: Input prompt (drug analysis request)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream output (NOT YET IMPLEMENTED)
            **kwargs: Additional generation parameters
            
        Returns:
            DrugDiscoveryOutput with generated text and metadata
        """
        # Format prompt if not already in chat format
        if "<|im_start|>" not in prompt:
            prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MODEL_CONFIG.get("max_length", 2048) - (max_new_tokens or 1024),
        ).to(self.device)
        
        # Set generation config
        gen_config = self.generation_config
        if max_new_tokens:
            gen_config.max_new_tokens = max_new_tokens
        if temperature:
            gen_config.temperature = temperature
        if top_p:
            gen_config.top_p = top_p
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return DrugDiscoveryOutput(
            generated_text=generated_text.strip(),
            input_text=prompt,
            generation_tokens=outputs.shape[1] - inputs["input_ids"].shape[1],
            model_name=self.model_name,
        )
    
    def analyze_drug(self, smiles: str, name: str = None) -> str:
        """
        Analyze a drug candidate given its SMILES string.
        
        Args:
            smiles: SMILES molecular representation
            name: Optional drug name
            
        Returns:
            Analysis report as string
        """
        prompt = f"Analyze this drug candidate and predict its approval likelihood:\nSMILES: {smiles}"
        if name:
            prompt += f"\nDrug Name: {name}"
        
        output = self.generate(prompt)
        return output.generated_text
    
    def explain_failure(self, smiles: str, name: str, failure_reason: str = None) -> str:
        """
        Explain why a drug failed in development.
        
        Args:
            smiles: SMILES molecular representation
            name: Drug name
            failure_reason: Known failure reason (optional)
            
        Returns:
            Failure analysis as string
        """
        prompt = f"This drug failed in clinical development. Explain why:\nDrug: {name}\nSMILES: {smiles}"
        if failure_reason:
            prompt += f"\nKnown issue: {failure_reason}"
        
        output = self.generate(prompt)
        return output.generated_text
    
    def suggest_improvements(self, smiles: str, name: str, problem: str) -> str:
        """
        Suggest structural modifications to improve a drug.
        
        Args:
            smiles: SMILES molecular representation
            name: Drug name
            problem: The problem to address
            
        Returns:
            Improvement suggestions as string
        """
        prompt = f"This drug failed due to {problem}. Suggest structural modifications to improve safety:\n{smiles}\nName: {name}"
        
        output = self.generate(prompt)
        return output.generated_text
    
    def compare_drugs(self, drug1_smiles: str, drug1_name: str, drug2_smiles: str, drug2_name: str) -> str:
        """
        Compare two drug candidates.
        
        Args:
            drug1_smiles: First drug SMILES
            drug1_name: First drug name
            drug2_smiles: Second drug SMILES
            drug2_name: Second drug name
            
        Returns:
            Comparison analysis as string
        """
        prompt = f"""Compare the safety profiles of these two drugs:
Drug 1: {drug1_smiles} ({drug1_name})
Drug 2: {drug2_smiles} ({drug2_name})"""
        
        output = self.generate(prompt)
        return output.generated_text
    
    def save_model(self, save_path: str):
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {save_path}...")
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print("Model saved successfully!")
    
    @classmethod
    def load_model(cls, load_path: str, **kwargs) -> "DrugDiscoveryLLM":
        """Load model from saved checkpoint."""
        print(f"Loading model from {load_path}...")
        return cls(model_name=load_path, **kwargs)


def get_tokenizer(model_name: str = None) -> AutoTokenizer:
    """
    Get tokenizer for the model.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Tokenizer instance
    """
    model_name = model_name or MODEL_CONFIG["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def estimate_memory_usage(model_params_billions: float = 14, batch_size: int = 4, seq_length: int = 2048) -> Dict:
    """
    Estimate GPU memory usage for training.
    
    Args:
        model_params_billions: Model parameters in billions
        batch_size: Training batch size
        seq_length: Sequence length
        
    Returns:
        Dict with memory estimates
    """
    # Rough estimates for bf16 training
    bytes_per_param = 2  # bf16 = 2 bytes
    
    # Model weights
    model_memory_gb = (model_params_billions * 1e9 * bytes_per_param) / 1024**3
    
    # Gradients (same size as weights)
    gradient_memory_gb = model_memory_gb
    
    # Optimizer states (AdamW = 2x model size)
    optimizer_memory_gb = 2 * model_memory_gb
    
    # Activations (rough estimate)
    # With gradient checkpointing, much lower than without
    activation_memory_gb = 0.5 * batch_size * seq_length * 4096 * bytes_per_param / 1024**3
    
    total_gb = model_memory_gb + gradient_memory_gb + optimizer_memory_gb + activation_memory_gb
    
    return {
        "model_gb": model_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_gb": total_gb,
        "recommended_vram_gb": total_gb * 1.2,  # 20% headroom
    }


if __name__ == "__main__":
    # Print memory estimates
    print("Memory Estimation for Qwen2.5-14B Full Fine-tuning:")
    estimates = estimate_memory_usage(14, batch_size=4, seq_length=2048)
    for key, value in estimates.items():
        print(f"  {key}: {value:.1f} GB")
    
    print(f"\n192GB VRAM is {'sufficient' if estimates['total_gb'] < 192 else 'insufficient'}")
    
    # Test model loading (uncomment to test)
    # print("\nTesting model loading...")
    # model = DrugDiscoveryLLM()
    # result = model.analyze_drug("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin")
    # print(f"\nAnalysis:\n{result}")
