#!/usr/bin/env python
"""
Fine-tuning script for Llama 3.3 models.
"""

import os
import sys
import logging
import argparse
from datasets import load_dataset

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fine_tuning import LlamaFineTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetuning.log')
    ]
)
logger = logging.getLogger('finetuning')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune Llama model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/finetune_config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (-1 = no distributed training)'
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
        help='Output directory for model (overrides config)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Initialize the fine-tuner
    logger.info(f"Initializing fine-tuner with config from {args.config}")
    fine_tuner = LlamaFineTuner(args.config, args.local_rank)
    
    # Override config settings if provided in args
    if args.output_dir:
        fine_tuner.output_dir = args.output_dir
        logger.info(f"Output directory overridden to {args.output_dir}")
    
    # Get data paths from config if not overridden
    train_file = args.train_file or fine_tuner.data_config.get('train_file')
    validation_file = args.validation_file or fine_tuner.data_config.get('validation_file')
    
    if not train_file:
        logger.error("No training file specified. Aborting.")
        sys.exit(1)
    
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
    
    # Prepare datasets for fine-tuning
    logger.info("Preparing datasets for fine-tuning")
    tokenized_datasets = fine_tuner.prepare_dataset(
        raw_datasets, 
        text_column=args.text_column
    )
    
    # Extract train and validation datasets
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else None
    
    # Print dataset information
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    
    # Run fine-tuning
    logger.info("Starting fine-tuning process")
    try:
        train_result = fine_tuner.train(train_dataset, eval_dataset)
        logger.info(f"Training completed. Results: {train_result}")
        
        # Create model directory if it doesn't exist
        os.makedirs(fine_tuner.output_dir, exist_ok=True)
        
        # Generate a sample for verification
        if eval_dataset:
            logger.info("Generating sample outputs for validation")
            sample_rows = eval_dataset.select(range(min(3, len(eval_dataset))))
            sample_texts = [row[args.text_column] for row in sample_rows]
            sample_outputs = fine_tuner.generate(sample_texts)
            
            for i, (prompt, output) in enumerate(zip(sample_texts, sample_outputs)):
                logger.info(f"Sample {i+1}:")
                logger.info(f"Prompt: {prompt[:100]}...")
                logger.info(f"Output: {output[:100]}...")
        
        logger.info(f"Fine-tuned model saved to {fine_tuner.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Fine-tuning process completed successfully")

if __name__ == "__main__":
    main()
