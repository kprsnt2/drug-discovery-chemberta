"""
Training Utilities for Drug Discovery Model

Includes:
- Learning rate scheduler with warmup
- Early stopping
- Checkpoint management
- Training metrics tracking
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, List, Callable
import math
import time
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import TRAINING_CONFIG, CHECKPOINT_DIR


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr_multiplier = self._get_lr_multiplier()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_multiplier
    
    def _get_lr_multiplier(self) -> float:
        """Calculate learning rate multiplier."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """
    Track training and validation metrics.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def update(self, metrics: Dict[str, float]):
        """Add metrics for current epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Track best
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            self.best_epoch = len(self.history['val_loss'])
    
    def save(self, path: str):
        """Save metrics history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str):
        """Load metrics history from JSON."""
        with open(path, 'r') as f:
            self.history = json.load(f)


class CheckpointManager:
    """
    Manage model checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = None,
        max_checkpoints: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Best model saved (val_loss: {metrics.get('val_loss', 'N/A'):.4f})")
        
        # Remove old checkpoints
        self._cleanup()
        
        print(f"  Checkpoint saved: {path.name}")
    
    def load(self, path: str = None) -> Dict:
        """Load checkpoint."""
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = torch.load(path)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def _cleanup(self):
        """Remove old checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device to train on
        scaler: Gradient scaler for mixed precision
        gradient_accumulation_steps: Gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Dict with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (step + 1),
            'acc': correct / total
        })
    
    return {
        'train_loss': total_loss / len(dataloader),
        'train_acc': correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation DataLoader
        device: Device to evaluate on
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(input_ids, attention_mask, labels)
        
        total_loss += outputs['loss'].item()
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    return {
        'val_loss': total_loss / len(dataloader),
        'val_acc': correct / total,
        'predictions': all_preds,
        'labels': all_labels
    }


def get_optimizer(model: nn.Module, learning_rate: float = None, weight_decay: float = None):
    """
    Create optimizer with weight decay excluding bias and LayerNorm.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay
    
    Returns:
        AdamW optimizer
    """
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    weight_decay = weight_decay or TRAINING_CONFIG['weight_decay']
    
    # Separate parameters for weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=TRAINING_CONFIG['adam_epsilon']
    )
    
    return optimizer
