#!/usr/bin/env python
"""
Run memory-efficient QLoRA fine-tuning for Llama 3.3 models.

This script demonstrates how to use the advanced QLoRA optimization for
memory-constrained environments.
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.memory_efficient_trainer import MemoryEfficientTrainer
from src.utils.memory_utils import print_gpu_memory_summary, clean_memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run memory-efficient QLoRA fine-tuning")
    
    # Model parameters
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="meta-llama/Llama-3.3-8B",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    
    # QLoRA parameters
    parser.add_argument(
        "--bits", 
        type=int, 
        default=4,
        choices=[4, 8, 16], 
        help="Quantization bits (4, 8, or 16)"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="wikitext",
        help="Dataset name from HuggingFace datasets hub"
    )
    parser.add_argument(
        "--dataset_config_name", 
        type=str, 
        default="wikitext-2-raw-v1",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--text_column", 
        type=str, 
        default="text",
        help="Column name for text data"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=2048,
        help="Maximum sequence length for training"
    )
    
    # Training parameters
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size (None for auto-detection)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=None,
        help="Gradient accumulation steps (None for auto-detection)"
    )
    
    # Memory optimization parameters
    parser.add_argument(
        "--auto_find_batch_size", 
        action="store_true",
        help="Automatically find optimal batch size"
    )
    parser.add_argument(
        "--auto_find_parameters", 
        action="store_true",
        help="Automatically find optimal parameters"
    )
    parser.add_argument(
        "--cpu_offloading", 
        action="store_true",
        help="Offload optimizer states to CPU"
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Use gradient checkpointing"
    )
    parser.add_argument(
        "--memory_profiling", 
        action="store_true",
        help="Enable memory profiling"
    )
    
    return parser.parse_args()


def main():
    """Main function to run memory-efficient training."""
    args = parse_args()
    
    # Print initial memory information
    if torch.cuda.is_available():
        logger.info("Initial GPU memory state:")
        print_gpu_memory_summary()
        clean_memory()
    else:
        logger.warning("CUDA not available, running on CPU only")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}/{args.dataset_config_name}")
    datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    # Split dataset for training and evaluation
    if "validation" in datasets:
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]
    else:
        # Split train set if no validation set is provided
        datasets = datasets["train"].train_test_split(test_size=0.1)
        train_dataset = datasets["train"]
        eval_dataset = datasets["test"]
    
    # Initialize memory-efficient trainer
    trainer = MemoryEfficientTrainer(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        bits=args.bits,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        memory_profiling=args.memory_profiling or True,  # Enable by default
        optimize_for_memory=True,
        auto_find_batch_size=args.auto_find_batch_size or True,  # Enable by default
        auto_find_parameters=args.auto_find_parameters or True,  # Enable by default
    )
    
    # Override settings if specified
    if args.cpu_offloading:
        trainer.use_cpu_offloading = True
    if args.gradient_checkpointing:
        trainer.use_gradient_checkpointing = True
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer with memory optimizations")
    model, tokenizer = trainer.load_model_and_tokenizer()
    
    # Prepare datasets
    logger.info("Preparing datasets for training")
    train_tokenized, eval_tokenized = trainer.prepare_training_data(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        text_column=args.text_column,
        max_seq_length=args.max_seq_length
    )
    
    # Create trainer
    logger.info("Creating memory-efficient trainer")
    trainer.create_trainer(
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Run training
    logger.info("Starting memory-efficient fine-tuning")
    trainer.train()
    
    # Run evaluation
    logger.info("Running evaluation")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Save hyperparameters
    trainer.save_hyperparameters()
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")
    
    # Print final memory information
    if torch.cuda.is_available():
        logger.info("Final GPU memory state:")
        print_gpu_memory_summary()


if __name__ == "__main__":
    main()
