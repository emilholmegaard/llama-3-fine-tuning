#!/usr/bin/env python
"""
Script for running hyperparameter optimization for Llama 3.3 fine-tuning.
"""

import os
import sys
import logging
import argparse
import json
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
        logging.FileHandler('hyperopt.log')
    ]
)
logger = logging.getLogger('hyperopt')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Llama fine-tuning')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/finetune_config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        choices=['optuna', 'ray'],
        default='optuna',
        help='Hyperparameter optimization backend (optuna or ray)'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=10,
        help='Number of optimization trials to run'
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
        help='Output directory for optimization results'
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
        help='Optimization direction'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout for the optimization in seconds'
    )
    
    parser.add_argument(
        '--ray_address',
        type=str,
        default=None,
        help='Address of Ray cluster for distributed optimization'
    )
    
    parser.add_argument(
        '--search_space',
        type=str,
        default=None,
        help='Path to JSON file defining the search space'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for hyperparameter optimization."""
    # Parse arguments
    args = parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data paths from config if not overridden
    train_file = args.train_file or config['data'].get('train_file')
    validation_file = args.validation_file or config['data'].get('validation_file')
    
    if not train_file:
        logger.error("No training file specified. Aborting.")
        sys.exit(1)
    
    if not validation_file:
        logger.warning("No validation file specified. Using train file for validation.")
        validation_file = train_file
    
    # Define output directory
    output_dir = args.output_dir or os.path.join(
        config['model']['output_dir'],
        'hyperopt'
    )
    
    # Initialize hyperparameter optimizer
    logger.info(f"Initializing hyperparameter optimizer with backend: {args.backend}")
    optimizer = HyperparameterOptimizer(
        base_config_path=args.config,
        backend=args.backend,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name="llama-lora-optimization",
        direction=args.direction,
        metric=args.metric,
        seed=config['training'].get('seed', 42),
        ray_address=args.ray_address
    )
    
    # Load custom search space if provided
    if args.search_space:
        with open(args.search_space, 'r') as f:
            search_space = json.load(f)
        optimizer.set_search_space(search_space)
        logger.info(f"Loaded custom search space from {args.search_space}")
    
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
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    best_params = optimizer.optimize(
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets.get("validation"),
        optimization_output_dir=output_dir
    )
    
    logger.info(f"Hyperparameter optimization completed")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
