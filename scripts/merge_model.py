#!/usr/bin/env python
"""
Script for merging LoRA adapters with base models or stacking multiple adapters.
"""

import os
import sys
import logging
import argparse
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_merging import ModelMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_merge.log')
    ]
)
logger = logging.getLogger('model_merger')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Merge LoRA adapters with base models or stack multiple adapters')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Merge adapter to base model
    merge_parser = subparsers.add_parser('merge', help='Merge adapter with base model')
    merge_parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='Base model name or path'
    )
    merge_parser.add_argument(
        '--adapter_path',
        type=str,
        required=True,
        help='Path to LoRA adapter'
    )
    merge_parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save merged model'
    )
    merge_parser.add_argument(
        '--adapter_name',
        type=str,
        default='default',
        help='Name of the adapter'
    )
    merge_parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16', 'fp32'],
        default='fp16',
        help='Precision to use'
    )
    merge_parser.add_argument(
        '--save_precision',
        type=str,
        choices=['fp16', 'bf16', 'fp32'],
        default=None,
        help='Precision to save (defaults to loaded precision)'
    )
    merge_parser.add_argument(
        '--quantize',
        action='store_true',
        help='Quantize the merged model'
    )
    merge_parser.add_argument(
        '--quantization_bits',
        type=int,
        choices=[4, 8],
        default=4,
        help='Bits to use for quantization'
    )
    
    # Stack multiple adapters
    stack_parser = subparsers.add_parser('stack', help='Stack multiple adapters')
    stack_parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='Base model name or path'
    )
    stack_parser.add_argument(
        '--adapter_paths',
        type=str,
        nargs='+',
        required=True,
        help='Paths to LoRA adapters to stack'
    )
    stack_parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save stacked adapter'
    )
    stack_parser.add_argument(
        '--adapter_weights',
        type=float,
        nargs='+',
        help='Weights for each adapter (must sum to 1.0)'
    )
    stack_parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16', 'fp32'],
        default='fp16',
        help='Precision to use'
    )
    
    # Quantize a merged model
    quantize_parser = subparsers.add_parser('quantize', help='Quantize a merged model')
    quantize_parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model to quantize'
    )
    quantize_parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save the quantized model'
    )
    quantize_parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8],
        default=4,
        help='Bits to use for quantization'
    )
    quantize_parser.add_argument(
        '--quant_type',
        type=str,
        choices=['nf4', 'fp4'],
        default='nf4',
        help='Quantization type to use'
    )
    quantize_parser.add_argument(
        '--double_quant',
        action='store_true',
        help='Use double quantization'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for model merging and stacking."""
    # Parse arguments
    args = parse_args()
    
    if args.command == 'merge':
        # Merge adapter to base model
        logger.info(f"Merging adapter {args.adapter_path} with base model {args.base_model}")
        
        # Initialize model merger
        merger = ModelMerger(
            base_model_name_or_path=args.base_model,
            precision=args.precision
        )
        
        # Create quantization config if requested
        quantization_config = None
        if args.quantize:
            quantization_config = {
                'bits': args.quantization_bits,
                'double_quant': True,
                'quant_type': 'nf4'
            }
        
        # Merge adapter with base model
        merged_model_path = merger.merge_lora_to_base_model(
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            adapter_name=args.adapter_name,
            save_precision=args.save_precision,
            quantization_config=quantization_config
        )
        
        logger.info(f"Merged model saved to {merged_model_path}")
        print(f"\nMerged model saved to {merged_model_path}")
        
    elif args.command == 'stack':
        # Stack multiple adapters
        logger.info(f"Stacking {len(args.adapter_paths)} adapters")
        
        # Initialize model merger
        merger = ModelMerger(
            base_model_name_or_path=args.base_model,
            precision=args.precision
        )
        
        # Validate adapter weights if provided
        adapter_weights = None
        if args.adapter_weights:
            if len(args.adapter_weights) != len(args.adapter_paths):
                logger.error("Number of adapter weights must match number of adapters")
                sys.exit(1)
            
            if abs(sum(args.adapter_weights) - 1.0) > 1e-6:
                logger.error("Adapter weights must sum to 1.0")
                sys.exit(1)
            
            adapter_weights = args.adapter_weights
        
        # Stack adapters
        stacked_adapter_path = merger.stack_adapters(
            adapter_paths=args.adapter_paths,
            output_path=args.output_path,
            adapter_weights=adapter_weights
        )
        
        logger.info(f"Stacked adapter saved to {stacked_adapter_path}")
        print(f"\nStacked adapter saved to {stacked_adapter_path}")
        
    elif args.command == 'quantize':
        # Quantize a merged model
        logger.info(f"Quantizing model {args.model_path}")
        
        # Initialize model merger
        merger = ModelMerger(
            base_model_name_or_path=args.model_path
        )
        
        # Create quantization config
        quantization_config = {
            'bits': args.bits,
            'double_quant': args.double_quant,
            'quant_type': args.quant_type
        }
        
        # Quantize model
        quantized_model_path = merger.quantize_merged_model(
            model_path=args.model_path,
            output_path=args.output_path,
            quantization_config=quantization_config
        )
        
        logger.info(f"Quantized model saved to {quantized_model_path}")
        print(f"\nQuantized model saved to {quantized_model_path}")
        
    else:
        logger.error("No command specified")
        sys.exit(1)
    
    logger.info("Model operation completed successfully")

if __name__ == "__main__":
    main()
