"""
Model Wrapper for Drug Discovery

ChemBERTa model with classification head and memory optimizations.
Includes gradient checkpointing support for fitting in 6GB VRAM.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG


class DrugDiscoveryModel(nn.Module):
    """
    ChemBERTa-based model for drug classification.
    
    Features:
    - Full finetuning (all parameters trainable)
    - Gradient checkpointing for memory efficiency
    - Classification head for binary prediction
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        dropout: Dropout probability
        use_gradient_checkpointing: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        model_name: str = None,
        num_labels: int = None,
        dropout: float = None,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name or MODEL_CONFIG['model_name']
        self.num_labels = num_labels or MODEL_CONFIG['num_labels']
        self.dropout_prob = dropout or MODEL_CONFIG['hidden_dropout_prob']
        
        # Load pretrained config
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.hidden_dropout_prob = self.dropout_prob
        self.config.attention_probs_dropout_prob = MODEL_CONFIG['attention_probs_dropout_prob']
        
        # Load pretrained model
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            config=self.config
        )
        
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (saves ~40% GPU memory)")
        
        # Classification head
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel: {self.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Hidden size: {self.config.hidden_size}")
        print(f"Number of layers: {self.config.num_hidden_layers}")
        print(f"Number of labels: {self.num_labels}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Optional labels for loss computation
        
        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Encode SMILES
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool: use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result['loss'] = loss_fn(logits, labels)
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        return preds, probs
    
    def save_model(self, save_path: str):
        """Save model weights and config."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), save_path / "model.pt")
        
        # Save config
        self.config.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, **kwargs):
        """Load model from saved weights."""
        load_path = Path(load_path)
        
        # Load config
        config = AutoConfig.from_pretrained(load_path)
        
        # Create model
        model = cls(
            model_name=kwargs.get('model_name', MODEL_CONFIG['model_name']),
            num_labels=kwargs.get('num_labels', MODEL_CONFIG['num_labels']),
            use_gradient_checkpointing=False  # Not needed for inference
        )
        
        # Load weights
        model.load_state_dict(torch.load(load_path / "model.pt"))
        
        print(f"Model loaded from {load_path}")
        return model


def get_tokenizer(model_name: str = None) -> AutoTokenizer:
    """
    Get tokenizer for the model.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Tokenizer instance
    """
    model_name = model_name or MODEL_CONFIG['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def estimate_memory_usage(model: nn.Module, batch_size: int = 8, seq_length: int = 128) -> dict:
    """
    Estimate GPU memory usage for training.
    
    Args:
        model: PyTorch model
        batch_size: Training batch size
        seq_length: Sequence length
    
    Returns:
        Dict with memory estimates in MB
    """
    # Parameter memory (FP16)
    param_memory = sum(p.numel() * 2 for p in model.parameters()) / 1024**2
    
    # Optimizer memory (AdamW stores 2 states per param)
    optimizer_memory = param_memory * 2
    
    # Gradient memory
    gradient_memory = param_memory
    
    # Activation memory (rough estimate with gradient checkpointing)
    # With checkpointing, we only store sqrt(L) activations
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    # Activations per layer (rough)
    activation_per_layer = batch_size * seq_length * hidden_size * 2 / 1024**2
    # With checkpointing, store ~sqrt(num_layers) activations
    import math
    checkpointed_layers = math.ceil(math.sqrt(num_layers))
    activation_memory = activation_per_layer * checkpointed_layers
    
    total = param_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        'parameters_mb': param_memory,
        'optimizer_mb': optimizer_memory,
        'gradients_mb': gradient_memory,
        'activations_mb': activation_memory,
        'total_mb': total,
        'total_gb': total / 1024
    }


if __name__ == "__main__":
    # Test model loading
    print("Testing DrugDiscoveryModel...")
    
    model = DrugDiscoveryModel(use_gradient_checkpointing=True)
    
    # Estimate memory
    memory = estimate_memory_usage(model)
    print(f"\nEstimated GPU memory usage:")
    for key, value in memory.items():
        if 'mb' in key:
            print(f"  {key}: {value:.1f} MB")
        else:
            print(f"  {key}: {value:.2f} GB")
    
    # Test forward pass
    tokenizer = get_tokenizer()
    
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    encoding = tokenizer(
        test_smiles,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    
    print(f"\nTest SMILES: {test_smiles}")
    print(f"Logits: {outputs['logits']}")
    print(f"Prediction: {'Approved' if outputs['logits'].argmax() == 1 else 'Failed'}")
