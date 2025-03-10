#!/usr/bin/env python
"""
Evaluation script for fine-tuned Llama models.

This script provides a command-line interface for evaluating fine-tuned models
against test datasets, generating comprehensive evaluation reports.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datasets import load_dataset
from peft import PeftModel
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.llama_wrapper import LlamaWrapper
from src.evaluation.metrics import (
    calculate_perplexity,
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_exact_match,
    calculate_f1_score,
    calculate_accuracy,
    calculate_domain_specific_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger('model_evaluation')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Llama model')
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to fine-tuned model directory'
    )
    
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test dataset (jsonl, csv, or parquet)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/evaluation',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['perplexity'],
        choices=['perplexity', 'rouge', 'bleu', 'exact_match', 'accuracy', 'f1', 'domain_specific'],
        help='Evaluation metrics to calculate'
    )
    
    parser.add_argument(
        '--domain_type',
        type=str,
        choices=['qa', 'summarization', 'classification'],
        help='Domain type for domain-specific evaluation'
    )
    
    parser.add_argument(
        '--prompt_column',
        type=str,
        default='prompt',
        help='Column name for prompts/inputs in the test data'
    )
    
    parser.add_argument(
        '--response_column',
        type=str,
        default='response',
        help='Column name for target responses in the test data'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum number of new tokens to generate'
    )
    
    parser.add_argument(
        '--generate_examples',
        type=int,
        default=0,
        help='Number of examples to generate in the report'
    )
    
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Load model in 4-bit precision (useful for large models)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run evaluation on'
    )
    
    return parser.parse_args()


def load_model(model_path: str, quantize: bool = False, device: str = 'auto') -> LlamaWrapper:
    """
    Load the fine-tuned model.
    
    Args:
        model_path: Path to the model directory
        quantize: Whether to quantize the model to 4 bits
        device: Device to load the model on
        
    Returns:
        Loaded LlamaWrapper instance
    """
    logger.info(f"Loading model from {model_path}")
    
    quantization_config = None
    if quantize:
        quantization_config = {
            'bits': 4,
            'double_quant': True,
            'quant_type': 'nf4'
        }
    
    # Check if this is a PEFT/LoRA model
    is_peft_model = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
    
    if is_peft_model:
        # For PEFT models, we need to load the base model first, then the adapter
        logger.info("Detected PEFT/LoRA model")
        
        # Try to find the base model name from adapter config
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'meta-llama/Llama-3.3-8B')
        
        # Load base model with quantization if needed
        wrapper = LlamaWrapper(
            model_name_or_path=base_model_name,
            quantization_config=quantization_config,
            device_map=device
        )
        
        # Load the adapter
        wrapper.model = PeftModel.from_pretrained(
            wrapper.model,
            model_path,
            device_map=device
        )
    else:
        # Standard model loading
        wrapper = LlamaWrapper(
            model_name_or_path=model_path,
            quantization_config=quantization_config,
            device_map=device
        )
    
    # Set to evaluation mode
    wrapper.model.eval()
    logger.info(f"Model loaded successfully on {wrapper.device}")
    
    return wrapper


def load_test_data(file_path: str, prompt_column: str, response_column: str) -> pd.DataFrame:
    """
    Load the test dataset.
    
    Args:
        file_path: Path to the test data file
        prompt_column: Column name for prompts
        response_column: Column name for responses
        
    Returns:
        DataFrame containing the test data
    """
    logger.info(f"Loading test data from {file_path}")
    
    # Determine file extension
    extension = os.path.splitext(file_path)[1].lstrip('.')
    if extension not in ['csv', 'json', 'jsonl', 'parquet']:
        extension = 'json'  # Default to json format
    
    # Load dataset
    dataset = load_dataset(extension, data_files={'test': file_path})['test']
    
    # Validate columns exist
    available_columns = dataset.column_names
    if prompt_column not in available_columns or response_column not in available_columns:
        available_cols_str = ', '.join(available_columns)
        error_msg = f"Required columns not found. Available columns: {available_cols_str}"
        if prompt_column not in available_columns:
            error_msg += f"\nPrompt column '{prompt_column}' not found"
        if response_column not in available_columns:
            error_msg += f"\nResponse column '{response_column}' not found"
        raise ValueError(error_msg)
    
    # Convert to pandas for easier handling
    data = dataset.to_pandas()
    logger.info(f"Loaded {len(data)} test examples")
    
    return data


def evaluate_model(
    model: LlamaWrapper,
    test_data: pd.DataFrame,
    metrics: List[str],
    prompt_column: str,
    response_column: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    domain_type: Optional[str] = None,
    num_examples: int = 0
) -> Dict[str, Any]:
    """
    Evaluate the model on the test data.
    
    Args:
        model: Model to evaluate
        test_data: Test dataset
        metrics: List of metrics to calculate
        prompt_column: Column name for prompts
        response_column: Column name for responses
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum number of tokens to generate
        domain_type: Domain type for domain-specific evaluation
        num_examples: Number of examples to include in the report
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting model evaluation")
    results = {}
    prompts = test_data[prompt_column].tolist()
    references = test_data[response_column].tolist()
    
    # Generate predictions for the test set
    logger.info(f"Generating predictions for {len(prompts)} examples")
    predictions = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_predictions = model.generate(
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Lower temperature for evaluation
            do_sample=False  # Deterministic for evaluation
        )
        predictions.extend(batch_predictions)
        logger.info(f"Generated predictions for examples {i} to {min(i+batch_size, len(prompts))}")
    
    # Calculate requested metrics
    logger.info("Calculating evaluation metrics")
    for metric in metrics:
        if metric == 'perplexity':
            # Perplexity requires the model and tokenizer
            perplexity_results = calculate_perplexity(
                model.model,
                model.tokenizer,
                references,
                batch_size=batch_size,
                device=str(model.device)
            )
            results['perplexity'] = perplexity_results
            
        elif metric == 'rouge':
            rouge_results = calculate_rouge_scores(predictions, references)
            results['rouge'] = rouge_results
            
        elif metric == 'bleu':
            # BLEU requires list of references for each prediction
            references_for_bleu = [[ref] for ref in references]
            bleu_results = calculate_bleu_score(predictions, references_for_bleu)
            results['bleu'] = bleu_results
            
        elif metric == 'exact_match':
            exact_match_score = calculate_exact_match(predictions, references)
            results['exact_match'] = exact_match_score
            
        elif metric == 'accuracy':
            accuracy_score = calculate_accuracy(predictions, references, normalize_text_inputs=True)
            results['accuracy'] = accuracy_score
            
        elif metric == 'f1':
            f1_results = calculate_f1_score(predictions, references, normalize_text_inputs=True)
            results['f1'] = f1_results
            
        elif metric == 'domain_specific' and domain_type:
            domain_results = calculate_domain_specific_metrics(
                predictions,
                references,
                domain_type
            )
            results['domain_specific'] = domain_results
    
    # Add examples if requested
    if num_examples > 0:
        num_to_include = min(num_examples, len(prompts))
        examples = []
        
        for i in range(num_to_include):
            examples.append({
                'prompt': prompts[i],
                'reference': references[i],
                'prediction': predictions[i]
            })
        
        results['examples'] = examples
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, model_path: str) -> str:
    """
    Save evaluation results to files.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save results
        model_path: Path to the evaluated model
        
    Returns:
        Path to the saved report file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.normpath(model_path))
    eval_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save full results as JSON
    results_file = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate a summary report
    report_file = os.path.join(eval_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write(f"# Model Evaluation Report\n\n")
        f.write(f"**Model:** {model_path}\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Metrics Summary\n\n")
        
        # Perplexity
        if 'perplexity' in results:
            f.write("### Perplexity\n\n")
            f.write(f"- **Perplexity:** {results['perplexity']['perplexity']:.4f}\n")
            f.write(f"- **Average Loss:** {results['perplexity']['avg_loss']:.4f}\n")
            f.write(f"- **Total Tokens:** {results['perplexity']['total_tokens']}\n\n")
        
        # ROUGE
        if 'rouge' in results:
            f.write("### ROUGE Scores\n\n")
            f.write("| Metric | Precision | Recall | F1 |\n")
            f.write("|--------|-----------|--------|----|\n")
            
            for rouge_type, scores in results['rouge'].items():
                f.write(f"| {rouge_type} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['fmeasure']:.4f} |\n")
            f.write("\n")
        
        # BLEU
        if 'bleu' in results:
            f.write("### BLEU Score\n\n")
            f.write(f"- **BLEU:** {results['bleu']['bleu']:.4f}\n")
            f.write("- **N-gram Precisions:**\n")
            for i, p in enumerate(results['bleu']['precisions']):
                f.write(f"  - {i+1}-gram: {p:.4f}\n")
            f.write("\n")
        
        # Exact Match
        if 'exact_match' in results:
            f.write("### Exact Match\n\n")
            f.write(f"- **Score:** {results['exact_match']:.4f}\n\n")
        
        # Accuracy
        if 'accuracy' in results:
            f.write("### Accuracy\n\n")
            f.write(f"- **Score:** {results['accuracy']:.4f}\n\n")
        
        # F1
        if 'f1' in results:
            f.write("### F1 Score\n\n")
            f.write(f"- **F1:** {results['f1']['f1']:.4f}\n")
            f.write(f"- **Precision:** {results['f1']['precision']:.4f}\n")
            f.write(f"- **Recall:** {results['f1']['recall']:.4f}\n\n")
        
        # Domain Specific
        if 'domain_specific' in results:
            f.write("### Domain-Specific Metrics\n\n")
            for name, value in results['domain_specific'].items():
                if isinstance(value, dict):
                    f.write(f"- **{name}:**\n")
                    for subname, subvalue in value.items():
                        f.write(f"  - {subname}: {subvalue:.4f}\n")
                else:
                    f.write(f"- **{name}:** {value:.4f}\n")
            f.write("\n")
        
        # Examples
        if 'examples' in results:
            f.write("## Generation Examples\n\n")
            for i, example in enumerate(results['examples']):
                f.write(f"### Example {i+1}\n\n")
                f.write(f"**Prompt:**\n```\n{example['prompt']}\n```\n\n")
                f.write(f"**Reference:**\n```\n{example['reference']}\n```\n\n")
                f.write(f"**Prediction:**\n```\n{example['prediction']}\n```\n\n")
                f.write("---\n\n")
    
    logger.info(f"Evaluation results saved to {eval_dir}")
    return report_file


def main():
    """Main entry point for evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the model
        model = load_model(args.model_path, args.quantize, args.device)
        
        # Load test data
        test_data = load_test_data(args.test_data, args.prompt_column, args.response_column)
        
        # Evaluate the model
        evaluation_results = evaluate_model(
            model=model,
            test_data=test_data,
            metrics=args.metrics,
            prompt_column=args.prompt_column,
            response_column=args.response_column,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            domain_type=args.domain_type if 'domain_specific' in args.metrics else None,
            num_examples=args.generate_examples
        )
        
        # Save results
        report_file = save_results(evaluation_results, args.output_dir, args.model_path)
        logger.info(f"Evaluation completed successfully. Report saved to {report_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Evaluation process completed")


if __name__ == "__main__":
    main()
