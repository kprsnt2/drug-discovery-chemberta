"""
Main Training Script for Drug Discovery Model

Full finetuning of ChemBERTa for drug success prediction.
Optimized for RTX 3050 6GB VRAM with:
- Gradient checkpointing
- Mixed precision (FP16) training
- Gradient accumulation
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, TRAINING_CONFIG, PROCESSED_DATA_DIR, 
    CHECKPOINT_DIR, RESULTS_DIR, LOGGING_CONFIG
)
from src.model import DrugDiscoveryModel, get_tokenizer, estimate_memory_usage
from src.dataset import create_dataloaders
from src.trainer import (
    get_optimizer, WarmupCosineScheduler, EarlyStopping,
    MetricsTracker, CheckpointManager, train_epoch, evaluate
)

from torch.cuda.amp import GradScaler


def setup_device():
    """Set up training device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Enable TF32 for Ampere GPUs (RTX 30xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available. Training on CPU will be very slow!")
    
    return device


def main(args):
    """Main training function."""
    print("="*60)
    print("Drug Discovery Model Training")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Check for existing data
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not train_path.exists():
        print("\nError: Training data not found!")
        print("Please run 'python scripts/download_all.py' first.")
        return
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=TRAINING_CONFIG['dataloader_num_workers']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Class weights: {class_weights}")
    
    # Create model
    print("\n[3/5] Creating model...")
    model = DrugDiscoveryModel(
        use_gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing']
    )
    model = model.to(device)
    
    # Estimate memory usage
    memory = estimate_memory_usage(model, args.batch_size)
    print(f"\nEstimated GPU memory: {memory['total_gb']:.2f} GB")
    
    if memory['total_gb'] > 5.5:
        print("WARNING: Memory usage may exceed 6GB. Consider reducing batch size.")
    
    # Create optimizer and scheduler
    print("\n[4/5] Setting up optimizer and scheduler...")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate)
    
    total_steps = len(train_loader) * args.epochs // TRAINING_CONFIG['gradient_accumulation_steps']
    warmup_steps = int(total_steps * TRAINING_CONFIG['warmup_ratio'])
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Initialize training components
    scaler = GradScaler() if TRAINING_CONFIG['fp16'] else None
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['early_stopping_patience'],
        min_delta=TRAINING_CONFIG['early_stopping_threshold']
    )
    metrics_tracker = MetricsTracker()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(CHECKPOINT_DIR),
        max_checkpoints=TRAINING_CONFIG['save_total_limit']
    )
    
    # Training loop
    print("\n[5/5] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {args.batch_size * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  Mixed precision: {TRAINING_CONFIG['fp16']}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
            max_grad_norm=TRAINING_CONFIG['max_grad_norm']
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Combine metrics
        metrics = {
            **train_metrics,
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time': epoch_time
        }
        
        metrics_tracker.update(metrics)
        
        # Print epoch summary
        print(f"  Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_acc']:.4f}")
        print(f"  Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.4f}")
        print(f"  LR: {metrics['learning_rate']:.2e} | Time: {epoch_time:.1f}s")
        
        # Check for best model
        is_best = metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = metrics['val_loss']
        
        # Save checkpoint
        if epoch % args.save_every == 0 or is_best:
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )
        
        # Early stopping
        if early_stopping(metrics['val_loss']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # GPU memory info
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU Memory: {memory_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        print()
    
    # Final evaluation on test set
    print("\n" + "="*40)
    print("Final Evaluation on Test Set")
    print("="*40)
    
    # Load best model
    checkpoint = checkpoint_manager.load()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['val_acc']:.4f}")
    
    # Save final metrics
    metrics_tracker.save(str(RESULTS_DIR / "training_metrics.json"))
    
    # Save test predictions
    import pandas as pd
    test_results = pd.DataFrame({
        'label': test_metrics['labels'],
        'prediction': test_metrics['predictions']
    })
    test_results.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    
    print("\n" + "="*40)
    print("Training Complete!")
    print("="*40)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test accuracy: {test_metrics['val_acc']:.4f}")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train drug discovery model"
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
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use (for testing)"
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
        args.max_samples = 100
        print("TEST RUN MODE - Using minimal configuration")
    
    main(args)
