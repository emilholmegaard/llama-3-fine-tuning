#!/usr/bin/env python
"""
Script to evaluate fine-tuned models using the evaluation module.

This script provides a command-line interface for evaluating fine-tuned models
using various metrics and datasets.
"""

import os
import argparse
import logging
import yaml
import torch
from pathlib import Path

from src.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned language models")
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model or model identifier from HuggingFace"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data file in JSONL or CSV format"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: data/evaluation/model_name)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="prompt",
        help="Key for input texts in the dataset (default: prompt)"
    )
    parser.add_argument(
        "--reference_key",
        type=str,
        default="completion",
        help="Key for reference texts in the dataset (default: completion)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["perplexity", "rouge", "bleu", "exact_match", "f1"],
        help="Metrics to evaluate (default: perplexity rouge bleu exact_match f1)"
    )
    parser.add_argument(
        "--domain_type",
        type=str,
        default=None,
        help="Domain-specific metrics to include (qa, summarization, classification)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on (default: auto)"
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        nargs="*",
        default=[],
        help="Paths to other models to compare with"
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run the evaluation script."""
    args = parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Determine the model name from path
    model_name = os.path.basename(args.model_path)
    
    # Set up output directory
    output_dir = args.output_dir or os.path.join("data", "evaluation", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    logger.info(f"Initializing evaluator for model: {args.model_path}")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit,
        output_dir=output_dir
    )
    
    # Load dataset
    logger.info(f"Loading test data from {args.test_data}")
    file_ext = Path(args.test_data).suffix.lower()
    dataset = evaluator.load_dataset(
        data_files=args.test_data,
        input_key=args.input_key,
        reference_key=args.reference_key
    )
    
    # Run evaluations based on requested metrics
    results = {}
    
    if "perplexity" in args.metrics:
        logger.info("Evaluating perplexity")
        perplexity_results = evaluator.evaluate_perplexity(
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        results["perplexity"] = perplexity_results
    
    # Generate text for all other metrics
    text_based_metrics = [m for m in args.metrics if m != "perplexity"]
    if text_based_metrics:
        logger.info("Generating text for evaluation")
        
        # Get generation config from yaml config if available
        generation_config = config.get("generation", {})
        
        examples = evaluator.generate_text(
            num_examples=args.num_examples,
            generation_config=generation_config
        )
        
        logger.info("Calculating metrics")
        metric_results = evaluator.calculate_metrics(
            examples=examples,
            metrics=text_based_metrics,
            domain_type=args.domain_type
        )
        results.update(metric_results)
    
    # Create visualizations
    logger.info("Creating visualizations")
    evaluator.visualize_results()
    
    # Compare with other models if specified
    if args.compare_with:
        comparison_evaluators = []
        
        for model_path in args.compare_with:
            logger.info(f"Loading comparison model: {model_path}")
            try:
                comp_evaluator = ModelEvaluator(
                    model_path=model_path,
                    device=args.device,
                    load_in_8bit=args.load_8bit,
                    load_in_4bit=args.load_4bit
                )
                
                # Load the same dataset
                comp_evaluator.load_dataset(
                    data_files=args.test_data,
                    input_key=args.input_key,
                    reference_key=args.reference_key
                )
                
                # Run the same evaluations
                if "perplexity" in args.metrics:
                    comp_evaluator.evaluate_perplexity(
                        batch_size=args.batch_size,
                        max_length=args.max_length
                    )
                
                if text_based_metrics:
                    comp_examples = comp_evaluator.generate_text(
                        num_examples=args.num_examples,
                        generation_config=generation_config
                    )
                    
                    comp_evaluator.calculate_metrics(
                        examples=comp_examples,
                        metrics=text_based_metrics,
                        domain_type=args.domain_type
                    )
                
                comparison_evaluators.append(comp_evaluator)
                
            except Exception as e:
                logger.error(f"Error loading comparison model {model_path}: {e}")
        
        if comparison_evaluators:
            logger.info("Comparing models")
            evaluator.compare_models(comparison_evaluators)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
