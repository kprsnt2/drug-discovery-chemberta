"""
Evaluation Script for Drug Discovery Model

Computes comprehensive metrics on test set:
- Accuracy, F1, Precision, Recall
- ROC-AUC and PR-AUC
- Confusion matrix
- Per-class metrics
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, PROCESSED_DATA_DIR, CHECKPOINT_DIR, 
    RESULTS_DIR, EVAL_CONFIG
)
from src.model import DrugDiscoveryModel, get_tokenizer
from src.dataset import DrugDiscoveryDataset
from torch.utils.data import DataLoader


def load_model(model_path: str = None, device: torch.device = None):
    """Load trained model.
    
    Supports both:
    - PyTorch .pt format (from train.py)
    - HuggingFace format (from train_cloud.py)
    """
    if model_path is None:
        model_path = CHECKPOINT_DIR / "best_model.pt"
    else:
        model_path = Path(model_path)
    
    # Check if it's a HuggingFace format directory (from train_cloud.py)
    # HF format: directory containing config.json and model files
    is_hf_format = (
        model_path.is_dir() and (model_path / "config.json").exists()
    ) or (
        model_path.suffix == "" and model_path.is_dir()  # Directory without extension
    )
    
    if is_hf_format:
        # Load HuggingFace format model
        from transformers import AutoModelForSequenceClassification
        print(f"Loading HuggingFace format model from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        print("Loaded HuggingFace format model (from train_cloud.py)")
        return model
    else:
        # Load PyTorch .pt format
        model = DrugDiscoveryModel(use_gradient_checkpointing=False)
        
        # Use weights_only=False to avoid security check (trusted local files)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        
        return model


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get predictions and probabilities for all samples."""
    all_labels = []
    all_preds = []
    all_probs = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask)
        # Handle both HuggingFace models and custom models
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # HuggingFace format
        else:
            logits = outputs['logits']  # Custom model format
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1_score': f1_score(labels, preds, average='binary'),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary'),
        'roc_auc': roc_auc_score(labels, probs),
        'pr_auc': average_precision_score(labels, probs),
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True/False positives/negatives
    tn, fp, fn, tp = cm.ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Specificity and sensitivity
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Failed', 'Approved'],
        yticklabels=['Failed', 'Approved']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_roc_curve(labels, probs, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to: {output_path}")


def plot_precision_recall_curve(labels, probs, output_path):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    auc = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to: {output_path}")


def main(args):
    """Main evaluation function."""
    print("="*60)
    print("Drug Discovery Model Evaluation")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for model and data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print("\nError: Test data not found!")
        print("Please run 'python scripts/download_all.py' first.")
        return
    
    model_path = Path(args.model_path) if args.model_path else CHECKPOINT_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("\nFor local training: run 'python train.py' first.")
        print("For cloud training: use --model_path to specify the model location:")
        print("  python evaluate.py --model_path checkpoints/cloud_<model_name>_<timestamp>/final_model")
        print("\nList available cloud models with: ls checkpoints/cloud_*/final_model")
        return
    
    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(model_path, device)
    
    # Load test data
    print("\n[2/4] Loading test data...")
    tokenizer = get_tokenizer()
    test_dataset = DrugDiscoveryDataset(str(test_path), tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Get predictions
    print("\n[3/4] Computing predictions...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    
    # Compute metrics
    print("\n[4/4] Computing metrics...")
    metrics = compute_metrics(labels, preds, probs)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'label': labels,
        'prediction': preds,
        'probability': probs
    })
    predictions_path = RESULTS_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    
    # Generate plots
    if args.generate_plots:
        print("\nGenerating plots...")
        plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")
        plot_roc_curve(labels, probs, RESULTS_DIR / "roc_curve.png")
        plot_precision_recall_curve(labels, probs, RESULTS_DIR / "pr_curve.png")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Failed', 'Approved']))
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate drug discovery model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        default=True,
        help="Generate evaluation plots"
    )
    
    args = parser.parse_args()
    main(args)
