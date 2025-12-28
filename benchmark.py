"""
Benchmark Script for Drug Discovery Model

Compares pretrained vs finetuned model performance:
- Baseline comparisons
- Cross-validation results
- Performance visualizations
- Generates comprehensive report
"""

import os
import sys
import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, PROCESSED_DATA_DIR, CHECKPOINT_DIR,
    RESULTS_DIR, BENCHMARK_CONFIG
)
from src.model import DrugDiscoveryModel, get_tokenizer
from src.dataset import DrugDiscoveryDataset
from torch.utils.data import DataLoader, Subset


class BenchmarkRunner:
    """Run comprehensive benchmarks and comparisons."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = get_tokenizer()
        self.results = {}
        
    def benchmark_pretrained(self, test_dataset):
        """Benchmark pretrained model (no finetuning)."""
        print("\n" + "-"*40)
        print("Benchmarking Pretrained Model (No Finetuning)")
        print("-"*40)
        
        try:
            model = DrugDiscoveryModel(use_gradient_checkpointing=False)
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"  WARNING: Could not load pretrained model: {e}")
            print("  Skipping pretrained benchmark.")
            return None
        
        metrics = self._evaluate(model, test_dataset)
        self.results['pretrained'] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def benchmark_finetuned(self, test_dataset, model_path: str = None):
        """Benchmark finetuned model.
        
        Supports both:
        - PyTorch .pt format (from train.py)
        - HuggingFace format (from train_cloud.py)
        """
        print("\n" + "-"*40)
        print("Benchmarking Finetuned Model")
        print("-"*40)
        
        if model_path is None:
            model_path = CHECKPOINT_DIR / "best_model.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"  WARNING: Finetuned model not found at {model_path}")
            print("  Skipping finetuned benchmark.")
            print("  For local training: run train.py first.")
            print("  For cloud training: use --model_path checkpoints/cloud_*/final_model")
            return None
        
        # Check if it's HuggingFace format (from train_cloud.py)
        # HF format: directory containing config.json and model files
        is_hf_format = (
            model_path.is_dir() and (model_path / "config.json").exists()
        ) or (
            model_path.suffix == "" and model_path.is_dir()
        )
        
        if is_hf_format:
            from transformers import AutoModelForSequenceClassification
            print(f"  Loading HuggingFace format model from: {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )
            model = model.to(self.device)
            model.eval()
            # Set flag for HF model to use different evaluation
            model._is_hf_model = True
        else:
            model = DrugDiscoveryModel(use_gradient_checkpointing=False)
            # Use weights_only=False to avoid security check (trusted local files)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            model._is_hf_model = False
        
        metrics = self._evaluate(model, test_dataset)
        self.results['finetuned'] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def benchmark_random_baseline(self, test_dataset):
        """Benchmark random baseline for comparison."""
        print("\n" + "-"*40)
        print("Benchmarking Random Baseline")
        print("-"*40)
        
        labels = test_dataset.get_labels()
        labels = np.array(labels)
        
        # Random predictions
        np.random.seed(42)
        preds = np.random.randint(0, 2, size=len(labels))
        probs = np.random.rand(len(labels))
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_score': f1_score(labels, preds, average='binary', zero_division=0),
            'roc_auc': 0.5,  # Random baseline
            'pr_auc': np.mean(labels),  # Class prior
        }
        
        self.results['random'] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def benchmark_majority_baseline(self, test_dataset):
        """Benchmark majority class baseline."""
        print("\n" + "-"*40)
        print("Benchmarking Majority Class Baseline")
        print("-"*40)
        
        labels = test_dataset.get_labels()
        labels = np.array(labels)
        
        # Majority class prediction
        majority_class = 1 if np.mean(labels) >= 0.5 else 0
        preds = np.full(len(labels), majority_class)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_score': f1_score(labels, preds, average='binary', zero_division=0),
            'roc_auc': 0.5,
            'pr_auc': np.mean(labels) if majority_class == 1 else 1 - np.mean(labels),
        }
        
        self.results['majority'] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Majority class: {'Approved' if majority_class == 1 else 'Failed'}")
        
        return metrics
    
    @torch.no_grad()
    def _evaluate(self, model, dataset):
        """Evaluate model on dataset."""
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            # Handle both HuggingFace models and custom models
            if hasattr(outputs, 'logits'):
                logits = outputs.logits  # HuggingFace format
            else:
                logits = outputs['logits']  # Custom model format
            
            # Convert to float32 for numerical stability
            logits = logits.float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Get probability of positive class
            if probs.shape[-1] >= 2:
                pos_probs = probs[:, 1].cpu().numpy()
            else:
                pos_probs = probs[:, 0].cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(pos_probs)
        
        labels = np.array(all_labels)
        preds = np.array(all_preds)
        probs = np.array(all_probs)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_score': f1_score(labels, preds, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        }


def generate_comparison_plot(results: dict, output_path: str):
    """Generate comparison bar plot."""
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']
    models = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, model in enumerate(models):
        values = [results[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model.capitalize(), color=colors[i % len(colors)])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Pretrained vs Finetuned')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'F1 Score', 'ROC-AUC', 'PR-AUC'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def generate_improvement_plot(results: dict, output_path: str):
    """Generate improvement percentage plot."""
    if 'pretrained' not in results or 'finetuned' not in results:
        print("Skipping improvement plot - need both pretrained and finetuned results")
        return
    
    metrics = ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']
    improvements = []
    
    for m in metrics:
        pre = results['pretrained'].get(m, 0)
        fine = results['finetuned'].get(m, 0)
        if pre > 0:
            improvement = ((fine - pre) / pre) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]
    bars = ax.bar(metrics, improvements, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        va = 'bottom' if val >= 0 else 'top'
        offset = 1 if val >= 0 else -1
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
               f'{val:+.1f}%', ha='center', va=va, fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Finetuning Improvement Over Pretrained Model')
    ax.set_xticklabels(['Accuracy', 'F1 Score', 'ROC-AUC', 'PR-AUC'])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved improvement plot to: {output_path}")


def generate_report(results: dict, output_path: str):
    """Generate HTML benchmark report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Drug Discovery Model Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .highlight {{ background: #d4edda !important; font-weight: bold; }}
        .metric-card {{ display: inline-block; width: 22%; margin: 1%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; }}
        img {{ max-width: 100%; margin: 20px 0; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Drug Discovery Model Benchmark Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <div style="text-align: center;">
"""
    
    # Add metric cards for finetuned model
    if 'finetuned' in results:
        metrics = results['finetuned']
        html += f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['f1_score']*100:.1f}%</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['roc_auc']*100:.1f}%</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['pr_auc']*100:.1f}%</div>
                <div class="metric-label">PR-AUC</div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
            </tr>
"""
    
    # Find best values for highlighting
    best_values = {}
    for metric in ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']:
        best_values[metric] = max(r.get(metric, 0) for r in results.values())
    
    # Add table rows
    for model, metrics in results.items():
        html += f"<tr>"
        html += f"<td><strong>{model.capitalize()}</strong></td>"
        for metric in ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']:
            val = metrics.get(metric, 0)
            cls = 'highlight' if val == best_values[metric] and val > 0.5 else ''
            html += f"<td class='{cls}'>{val:.4f}</td>"
        html += "</tr>"
    
    html += """
        </table>
        
        <h2>Visualizations</h2>
        <img src="comparison_plot.png" alt="Model Comparison">
        <img src="improvement_plot.png" alt="Improvement Analysis">
        
        <h2>Interpretation</h2>
        <ul>
"""
    
    # Add interpretations
    if 'finetuned' in results and 'pretrained' in results:
        acc_imp = (results['finetuned']['accuracy'] - results['pretrained']['accuracy']) * 100
        auc_imp = (results['finetuned']['roc_auc'] - results['pretrained']['roc_auc']) * 100
        
        html += f"<li>Finetuning improved accuracy by <strong>{acc_imp:+.1f}%</strong></li>"
        html += f"<li>ROC-AUC increased by <strong>{auc_imp:+.1f}%</strong> after finetuning</li>"
        
        if results['finetuned']['roc_auc'] > 0.7:
            html += "<li>The model shows <strong>good discriminative ability</strong> (AUC > 0.7)</li>"
    
    html += """
        </ul>
        
        <h2>Methodology</h2>
        <p>
            This benchmark compares model performance on drug success prediction task:
            <ul>
                <li><strong>Pretrained</strong>: ChemBERTa model without any finetuning</li>
                <li><strong>Finetuned</strong>: ChemBERTa after full finetuning on drug dataset</li>
                <li><strong>Random</strong>: Random predictions baseline</li>
                <li><strong>Majority</strong>: Always predicting the majority class</li>
            </ul>
        </p>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Saved HTML report to: {output_path}")


def main(args):
    """Main benchmark function."""
    print("="*60)
    print("Drug Discovery Model Benchmarking")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for test data
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"\nError: Test data not found at {test_path}")
        print("Please run 'python scripts/download_all.py' first.")
        return
    
    # Load test dataset
    print("\nLoading test dataset...")
    tokenizer = get_tokenizer()
    test_dataset = DrugDiscoveryDataset(str(test_path), tokenizer)
    print(f"Test samples: {len(test_dataset)}")
    
    # Run benchmarks
    runner = BenchmarkRunner(device)
    
    # Baseline benchmarks
    runner.benchmark_random_baseline(test_dataset)
    runner.benchmark_majority_baseline(test_dataset)
    
    # Model benchmarks
    runner.benchmark_pretrained(test_dataset)
    
    model_path = args.model_path if args.model_path else None
    runner.benchmark_finetuned(test_dataset, model_path)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(runner.results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate plots
    if args.generate_report:
        print("\nGenerating visualizations...")
        generate_comparison_plot(runner.results, RESULTS_DIR / "comparison_plot.png")
        generate_improvement_plot(runner.results, RESULTS_DIR / "improvement_plot.png")
        generate_report(runner.results, RESULTS_DIR / "benchmark_report.html")
    
    # Print summary table
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    headers = ['Model', 'Accuracy', 'F1 Score', 'ROC-AUC', 'PR-AUC']
    print(f"{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'ROC-AUC':<12} {'PR-AUC':<12}")
    print("-"*60)
    
    for model, metrics in runner.results.items():
        print(f"{model.capitalize():<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f} "
              f"{metrics['pr_auc']:<12.4f}")
    
    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    if args.generate_report:
        print(f"Open {RESULTS_DIR / 'benchmark_report.html'} in a browser to view the report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark drug discovery model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to finetuned model checkpoint"
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        default=True,
        help="Generate HTML report and plots"
    )
    
    args = parser.parse_args()
    main(args)
