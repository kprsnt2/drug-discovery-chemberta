"""
Evaluation Script for Drug Discovery Text Generation Model

Generates sample outputs and evaluates quality on test set.
For text generation, we evaluate:
- Perplexity (how well model predicts test set)
- Sample generation quality (human review)
- Response consistency
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_CONFIG, PROCESSED_DATA_DIR, CHECKPOINT_DIR, 
    RESULTS_DIR, GENERATION_CONFIG
)
from src.model import DrugDiscoveryLLM, get_tokenizer

# Sample prompts for evaluation
EVAL_PROMPTS = [
    {
        "prompt": "Analyze this drug candidate and predict its approval likelihood:\nSMILES: CC(=O)OC1=CC=CC=C1C(=O)O\nDrug Name: Aspirin",
        "type": "drug_analysis",
        "expected_keywords": ["approved", "molecular", "properties", "safety"]
    },
    {
        "prompt": "This drug failed in clinical development. Explain why:\nDrug: Rofecoxib (Vioxx)\nSMILES: CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3\nKnown issue: Cardiovascular toxicity",
        "type": "failure_analysis",
        "expected_keywords": ["cardiovascular", "COX-2", "risk", "withdrawn"]
    },
    {
        "prompt": "Compare the safety profiles of these two drugs:\nDrug 1: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)\nDrug 2: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O (Ibuprofen)",
        "type": "comparison",
        "expected_keywords": ["both", "NSAIDs", "difference", "safety"]
    },
    {
        "prompt": "This drug failed due to hepatotoxicity. Suggest structural modifications to improve safety:\nCC1=C(C2=CC=CC=C2O1)CC(C)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4\nName: Troglitazone",
        "type": "improvement_suggestion",
        "expected_keywords": ["modification", "hepato", "reduce", "safer"]
    },
    {
        "prompt": "Describe the molecular properties and drug-likeness of:\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "type": "property_analysis",
        "expected_keywords": ["molecular weight", "LogP", "Lipinski", "oral"]
    },
]


def load_model(model_path: str = None, device: str = None):
    """Load trained text generation model."""
    if model_path is None:
        # Find latest checkpoint
        checkpoints = list(CHECKPOINT_DIR.glob("run_*/final"))
        if checkpoints:
            model_path = str(sorted(checkpoints)[-1])
        else:
            model_path = MODEL_CONFIG['model_name']
    
    print(f"Loading model from: {model_path}")
    model = DrugDiscoveryLLM(
        model_name=model_path,
        device=device,
        use_flash_attention=False,
    )
    return model


def evaluate_generation_quality(model, prompts: list, max_new_tokens: int = 512):
    """
    Evaluate generation quality on sample prompts.
    
    Returns dict with:
    - Generated responses
    - Response lengths
    - Keyword coverage
    """
    results = []
    
    print("\nGenerating responses for evaluation prompts...")
    for item in tqdm(prompts, desc="Generating"):
        prompt = item["prompt"]
        expected_keywords = item.get("expected_keywords", [])
        
        output = model.generate(prompt, max_new_tokens=max_new_tokens)
        response = output.generated_text
        
        # Check keyword coverage
        response_lower = response.lower()
        keywords_found = [kw for kw in expected_keywords if kw.lower() in response_lower]
        keyword_coverage = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
        
        results.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "prompt_type": item.get("type", "unknown"),
            "response": response,
            "response_length": len(response),
            "tokens_generated": output.generation_tokens,
            "expected_keywords": expected_keywords,
            "keywords_found": keywords_found,
            "keyword_coverage": keyword_coverage,
        })
    
    return results


def compute_metrics(results: list):
    """Compute aggregate metrics from evaluation results."""
    if not results:
        return {}
    
    avg_response_length = sum(r["response_length"] for r in results) / len(results)
    avg_tokens = sum(r["tokens_generated"] for r in results) / len(results)
    avg_keyword_coverage = sum(r["keyword_coverage"] for r in results) / len(results)
    
    # Check for empty/failed responses
    failed_responses = [r for r in results if r["response_length"] < 50]
    
    metrics = {
        "total_samples": len(results),
        "avg_response_length": avg_response_length,
        "avg_tokens_generated": avg_tokens,
        "avg_keyword_coverage": avg_keyword_coverage,
        "failed_responses": len(failed_responses),
        "success_rate": (len(results) - len(failed_responses)) / len(results),
    }
    
    # Per-type metrics
    types = set(r["prompt_type"] for r in results)
    for prompt_type in types:
        type_results = [r for r in results if r["prompt_type"] == prompt_type]
        metrics[f"{prompt_type}_count"] = len(type_results)
        metrics[f"{prompt_type}_avg_length"] = sum(r["response_length"] for r in type_results) / len(type_results)
    
    return metrics


def main(args):
    """Main evaluation function."""
    print("="*60)
    print("Drug Discovery Text Generation Model Evaluation")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print("\n[1/3] Loading model...")
    model = load_model(args.model_path, device)
    
    # Evaluate generation quality
    print("\n[2/3] Evaluating generation quality...")
    results = evaluate_generation_quality(
        model, 
        EVAL_PROMPTS,
        max_new_tokens=args.max_tokens
    )
    
    # Compute metrics
    print("\n[3/3] Computing metrics...")
    metrics = compute_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples evaluated: {metrics['total_samples']}")
    print(f"Average response length: {metrics['avg_response_length']:.0f} chars")
    print(f"Average tokens generated: {metrics['avg_tokens_generated']:.0f}")
    print(f"Keyword coverage: {metrics['avg_keyword_coverage']*100:.1f}%")
    print(f"Success rate: {metrics['success_rate']*100:.1f}%")
    
    # Print sample outputs
    print("\n" + "-"*60)
    print("SAMPLE OUTPUTS")
    print("-"*60)
    for i, result in enumerate(results[:3], 1):
        print(f"\n[Sample {i}] {result['prompt_type'].upper()}")
        print(f"Prompt: {result['prompt']}")
        print(f"Response ({result['response_length']} chars):")
        print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
        print(f"Keywords found: {result['keywords_found']}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = RESULTS_DIR / "generation_eval_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save full results
    results_path = RESULTS_DIR / "generation_eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate drug discovery text generation model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: latest in checkpoints/)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response"
    )
    
    args = parser.parse_args()
    main(args)
