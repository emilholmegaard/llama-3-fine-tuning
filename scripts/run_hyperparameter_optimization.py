#!/usr/bin/env python
"""
Hyperparameter optimization script for Llama LoRA fine-tuning.
"""

import os
import sys
import logging
import argparse
import yaml
from datasets import load_dataset

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hyperparameter_optimization import HyperparameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hyperparameter_optimization.log')
    ]
)
logger = logging.getLogger('hyperopt')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Llama LoRA fine-tuning')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/finetune_config.yaml',
        help='Path to base configuration YAML file'
    )
    
    parser.add_argument(
        '--search_space', 
        type=str, 
        default=None,
        help='Path to search space YAML file'
    )
    
    parser.add_argument(
        '--backend', 
        type=str, 
        choices=['optuna', 'ray'],
        default='optuna',
        help='Hyperparameter optimization backend'
    )
    
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=10,
        help='Number of trials to run'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        help='Path to training data file (overrides config)'
    )
    
    parser.add_argument(
        '--validation_file',
        type=str,
        help='Path to validation data file (overrides config)'
    )
    
    parser.add_argument(
        '--text_column',
        type=str,
        default='text',
        help='Name of the text column in the dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for hyperparameter search results (overrides config)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='eval_loss',
        help='Metric to optimize'
    )
    
    parser.add_argument(
        '--direction',
        type=str,
        choices=['minimize', 'maximize'],
        default='minimize',
        help='Direction of optimization'
    )
    
    parser.add_argument(
        '--ray_address',
        type=str,
        default=None,
        help='Ray cluster address for distributed optimization'
    )
    
    parser.add_argument(
        '--gpus_per_trial',
        type=float,
        default=1.0,
        help='GPUs per trial (can be fractional)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for the entire optimization'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for hyperparameter optimization."""
    # Parse arguments
    args = parse_args()
    
    # Load base configuration
    logger.info(f"Loading base configuration from {args.config}")
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load search space if provided
    search_space = None
    if args.search_space:
        logger.info(f"Loading search space from {args.search_space}")
        with open(args.search_space, 'r') as f:
            search_space = yaml.safe_load(f)
    
    # Override output directory if provided
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(
            base_config.get('model', {}).get('output_dir', 'data/models/finetuned-model/'),
            'hpo_results'
        )
    
    # Get data paths from config if not overridden
    train_file = args.train_file or base_config.get('data', {}).get('train_file')
    validation_file = args.validation_file or base_config.get('data', {}).get('validation_file')
    
    if not train_file:
        logger.error("No training file specified. Aborting.")
        sys.exit(1)
    
    if not validation_file:
        logger.warning("No validation file specified. Using training data for evaluation.")
        validation_file = train_file
    
    # Initialize hyperparameter optimizer
    logger.info(f"Initializing hyperparameter optimizer with {args.backend} backend")
    optimizer = HyperparameterOptimizer(
        base_config_path=args.config,
        backend=args.backend,
        n_trials=args.n_trials,
        timeout=args.timeout,
        direction=args.direction,
        metric=args.metric,
        seed=args.seed,
        ray_address=args.ray_address,
        n_gpus_per_trial=args.gpus_per_trial
    )
    
    # Set custom search space if provided
    if search_space:
        optimizer.set_search_space(search_space)
    
    # Load datasets
    logger.info(f"Loading training dataset from {train_file}")
    data_files = {"train": train_file}
    if validation_file:
        logger.info(f"Loading validation dataset from {validation_file}")
        data_files["validation"] = validation_file
    
    # Determine file extension for dataset loading
    extension = os.path.splitext(train_file)[1].lstrip('.')
    if extension not in ['csv', 'json', 'jsonl', 'parquet']:
        extension = 'json'  # Default to json format
    
    # Load the datasets
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    # Run hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    try:
        best_params = optimizer.optimize(
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets.get("validation"),
            optimization_output_dir=output_dir
        )
        
        logger.info(f"Hyperparameter optimization completed. Best parameters: {best_params}")
        
        # Load best config
        best_config_path = os.path.join(output_dir, "best_config.yaml")
        if os.path.exists(best_config_path):
            logger.info(f"Best configuration saved to {best_config_path}")
            print(f"\nBest configuration saved to {best_config_path}")
            
            # Display best parameters
            print("\nBest hyperparameters:")
            for param_name, param_value in best_params.items():
                print(f"  {param_name}: {param_value}")
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Hyperparameter optimization process completed successfully")

if __name__ == "__main__":
    main()
