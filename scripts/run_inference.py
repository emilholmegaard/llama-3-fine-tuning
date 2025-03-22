#!/usr/bin/env python
"""
Inference Runner for Llama 3 fine-tuned models.

This script allows running inference with fine-tuned models on various input sources,
including single prompts, batch files, and interactive mode.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.llama_wrapper import LlamaWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger('inference_runner')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Llama models')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to fine-tuned model directory'
    )
    model_group.add_argument(
        '--quantize',
        action='store_true',
        help='Load model in 4-bit precision'
    )
    model_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--prompt',
        type=str,
        help='Single prompt for inference'
    )
    input_group.add_argument(
        '--input_file',
        type=str,
        help='File containing prompts (txt, csv, jsonl)'
    )
    input_group.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    # File format arguments
    file_group = parser.add_argument_group('File Options')
    file_group.add_argument(
        '--input_column',
        type=str,
        default='prompt',
        help='Column name for prompts in CSV/JSONL input'
    )
    file_group.add_argument(
        '--output_column',
        type=str,
        default='response',
        help='Column name for responses in CSV/JSONL output'
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group('Generation Options')
    gen_group.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate'
    )
    gen_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    gen_group.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling probability'
    )
    gen_group.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling'
    )
    gen_group.add_argument(
        '--do_sample',
        action='store_true',
        help='Use sampling instead of greedy decoding'
    )
    gen_group.add_argument(
        '--num_beams',
        type=int,
        default=1,
        help='Number of beams for beam search'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output_file',
        type=str,
        help='File to save inference results (defaults to input filename with _results suffix)'
    )
    output_group.add_argument(
        '--output_format',
        type=str,
        default=None,
        choices=['txt', 'csv', 'jsonl'],
        help='Output format (defaults to input format)'
    )
    output_group.add_argument(
        '--show_metrics',
        action='store_true',
        help='Show generation metrics (time, tokens/sec)'
    )
    output_group.add_argument(
        '--save_session',
        action='store_true',
        help='Save the interactive session to a file'
    )
    
    return parser.parse_args()


def load_model(model_path: str, quantize: bool = False, device: str = 'auto') -> LlamaWrapper:
    """
    Load a fine-tuned Llama model.
    
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
    
    try:
        if is_peft_model:
            # For PEFT models, load the base model first, then the adapter
            logger.info("Detected PEFT/LoRA model")
            
            # Try to find the base model name from adapter config
            adapter_config_path = os.path.join(model_path, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path', 'meta-llama/Llama-3.3-8B')
            
            logger.info(f"Loading base model: {base_model_name}")
            from peft import PeftModel
            
            # Load base model with quantization if needed
            wrapper = LlamaWrapper(
                model_name_or_path=base_model_name,
                quantization_config=quantization_config,
                device_map=device
            )
            
            # Load the adapter
            logger.info(f"Loading adapter from {model_path}")
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
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_prompts_from_file(file_path: str, input_column: str = 'prompt') -> List[str]:
    """
    Load prompts from a file.
    
    Args:
        file_path: Path to the file containing prompts
        input_column: Column name for prompts in CSV/JSONL files
        
    Returns:
        List of prompts
    """
    logger.info(f"Loading prompts from {file_path}")
    
    # Determine file extension
    extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if extension == '.txt':
            # For text files, each line is a prompt
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        
        elif extension == '.csv':
            # For CSV files, load into pandas
            df = pd.read_csv(file_path)
            
            if input_column not in df.columns:
                raise ValueError(f"Column '{input_column}' not found in CSV file. Available columns: {', '.join(df.columns)}")
            
            prompts = df[input_column].tolist()
        
        elif extension in ['.json', '.jsonl']:
            # For JSON/JSONL files
            if extension == '.json':
                # Single JSON object or array
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # JSON array of objects
                    if all(isinstance(item, dict) and input_column in item for item in data):
                        prompts = [item[input_column] for item in data]
                    else:
                        # Array of strings
                        prompts = data
                elif isinstance(data, dict) and input_column in data:
                    # Single object
                    prompts = [data[input_column]]
                else:
                    raise ValueError(f"JSON file does not contain expected format with '{input_column}' field")
            
            else:
                # JSONL: one JSON object per line
                prompts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            if input_column in item:
                                prompts.append(item[input_column])
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts
    
    except Exception as e:
        logger.error(f"Error loading prompts from file: {e}")
        raise


def run_inference(
    model: LlamaWrapper,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = False,
    num_beams: int = 1,
    show_metrics: bool = False
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Run inference on a list of prompts.
    
    Args:
        model: LlamaWrapper instance
        prompts: List of prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling
        do_sample: Whether to use sampling
        num_beams: Number of beams for beam search
        show_metrics: Whether to show generation metrics
        
    Returns:
        Tuple of (responses, metrics)
    """
    logger.info(f"Running inference on {len(prompts)} prompts")
    
    responses = []
    metrics = {
        'total_time': 0,
        'tokens_generated': 0,
        'tokens_per_second': 0
    }
    
    start_time = time.time()
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating responses")):
        try:
            prompt_start_time = time.time()
            
            # Run generation
            response = model.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams
            )[0]
            
            prompt_end_time = time.time()
            prompt_time = prompt_end_time - prompt_start_time
            
            # Estimate tokens generated (rough approximation)
            prompt_tokens = len(model.tokenizer.encode(prompt))
            response_tokens = len(model.tokenizer.encode(response))
            new_tokens = response_tokens - prompt_tokens
            
            # Update metrics
            metrics['tokens_generated'] += new_tokens
            metrics['total_time'] += prompt_time
            
            # Add response
            responses.append(response)
            
            if show_metrics:
                tokens_per_second = new_tokens / prompt_time if prompt_time > 0 else 0
                logger.info(f"Prompt {i+1}/{len(prompts)}: Generated {new_tokens} tokens in {prompt_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
        
        except Exception as e:
            logger.error(f"Error generating response for prompt {i+1}: {e}")
            responses.append(f"ERROR: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate final metrics
    if metrics['total_time'] > 0:
        metrics['tokens_per_second'] = metrics['tokens_generated'] / metrics['total_time']
    metrics['total_time'] = total_time
    
    if show_metrics:
        logger.info(f"Inference completed in {total_time:.2f}s")
        logger.info(f"Generated {metrics['tokens_generated']} tokens at {metrics['tokens_per_second']:.2f} tokens/s")
    
    return responses, metrics if show_metrics else None


def save_results(
    prompts: List[str],
    responses: List[str],
    output_file: str,
    output_format: str,
    metrics: Optional[Dict[str, float]] = None,
    input_column: str = 'prompt',
    output_column: str = 'response'
):
    """
    Save inference results to a file.
    
    Args:
        prompts: List of prompts
        responses: List of responses
        output_file: Path to save results
        output_format: Output format (txt, csv, jsonl)
        metrics: Optional generation metrics
        input_column: Column name for prompts
        output_column: Column name for responses
    """
    logger.info(f"Saving results to {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    try:
        if output_format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    f.write(f"Prompt {i+1}:\n{prompt}\n\n")
                    f.write(f"Response {i+1}:\n{response}\n\n")
                    f.write("-" * 80 + "\n\n")
                
                if metrics:
                    f.write("Generation Metrics:\n")
                    f.write(f"Total time: {metrics['total_time']:.2f}s\n")
                    f.write(f"Tokens generated: {metrics['tokens_generated']}\n")
                    f.write(f"Tokens per second: {metrics['tokens_per_second']:.2f}\n")
        
        elif output_format == 'csv':
            df = pd.DataFrame({
                input_column: prompts,
                output_column: responses
            })
            
            if metrics:
                # Add metrics as comments in the header
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Generation Metrics: ")
                    f.write(f"Total time: {metrics['total_time']:.2f}s, ")
                    f.write(f"Tokens generated: {metrics['tokens_generated']}, ")
                    f.write(f"Tokens per second: {metrics['tokens_per_second']:.2f}\n")
                
                # Append the CSV data
                df.to_csv(output_file, index=False, mode='a')
            else:
                df.to_csv(output_file, index=False)
        
        elif output_format == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write results
                for prompt, response in zip(prompts, responses):
                    item = {
                        input_column: prompt,
                        output_column: response,
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(json.dumps(item) + '\n')
                
                # Write metrics as a special record if available
                if metrics:
                    metrics_record = {
                        'type': 'metrics',
                        'total_time': metrics['total_time'],
                        'tokens_generated': metrics['tokens_generated'],
                        'tokens_per_second': metrics['tokens_per_second'],
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(json.dumps(metrics_record) + '\n')
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Results saved successfully to {output_file}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def run_interactive_mode(
    model: LlamaWrapper,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = False,
    num_beams: int = 1,
    save_session: bool = False,
    show_metrics: bool = False
):
    """
    Run the model in interactive mode.
    
    Args:
        model: LlamaWrapper instance
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling
        do_sample: Whether to use sampling
        num_beams: Number of beams for beam search
        save_session: Whether to save the session
        show_metrics: Whether to show generation metrics
    """
    logger.info("Starting interactive mode")
    print("\n" + "="*80)
    print("Llama 3 Interactive Mode")
    print("Type 'exit', 'quit', or press Ctrl+C to exit")
    print("="*80 + "\n")
    
    session = []
    
    try:
        while True:
            # Get user input
            prompt = input("\nYou: ")
            
            # Check if user wants to exit
            if prompt.lower() in ['exit', 'quit']:
                break
            
            # Skip empty prompts
            if not prompt.strip():
                continue
            
            try:
                # Generate response
                start_time = time.time()
                
                response = model.generate(
                    [prompt],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    num_beams=num_beams
                )[0]
                
                end_time = time.time()
                
                # Print response
                print(f"\nModel: {response}\n")
                
                # Show metrics if requested
                if show_metrics:
                    generation_time = end_time - start_time
                    
                    # Estimate tokens
                    prompt_tokens = len(model.tokenizer.encode(prompt))
                    response_tokens = len(model.tokenizer.encode(response))
                    tokens_generated = response_tokens - prompt_tokens
                    
                    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                    
                    print(f"[Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)]")
                
                # Save to session history
                session.append({
                    'prompt': prompt,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
            
            except KeyboardInterrupt:
                break
            
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print(f"\nError: {str(e)}\n")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    print("\nInteractive session ended")
    
    # Save session if requested
    if save_session and session:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = f"interactive_session_{timestamp}.jsonl"
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                for entry in session:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"Session saved to {session_file}")
            logger.info(f"Interactive session saved to {session_file}")
        
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            print(f"Error saving session: {str(e)}")


def determine_output_format(input_file: Optional[str], output_format: Optional[str], output_file: Optional[str]) -> Tuple[str, str]:
    """
    Determine the output format and file based on inputs.
    
    Args:
        input_file: Input file path
        output_format: Requested output format
        output_file: Requested output file path
        
    Returns:
        Tuple of (output_file, output_format)
    """
    # Default format is txt
    if not output_format:
        if input_file:
            # Use the same format as input
            input_ext = os.path.splitext(input_file)[1].lower()
            if input_ext == '.csv':
                output_format = 'csv'
            elif input_ext in ['.json', '.jsonl']:
                output_format = 'jsonl'
            else:
                output_format = 'txt'
        else:
            output_format = 'txt'
    
    # Default output file
    if not output_file:
        if input_file:
            # Use input file name with _results suffix
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_results.{output_format}"
        else:
            # Use a timestamped file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"inference_results_{timestamp}.{output_format}"
    
    return output_file, output_format


def main():
    """Main entry point for inference runner."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load model
        model = load_model(args.model_path, args.quantize, args.device)
        
        # Interactive mode
        if args.interactive:
            run_interactive_mode(
                model=model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                save_session=args.save_session,
                show_metrics=args.show_metrics
            )
            return
        
        # Prepare prompts
        if args.prompt:
            # Single prompt
            prompts = [args.prompt]
        else:
            # Load from file
            prompts = load_prompts_from_file(args.input_file, args.input_column)
        
        # Run inference
        responses, metrics = run_inference(
            model=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            show_metrics=args.show_metrics
        )
        
        # Determine output format and file
        output_file, output_format = determine_output_format(
            args.input_file, 
            args.output_format, 
            args.output_file
        )
        
        # Save results
        save_results(
            prompts=prompts,
            responses=responses,
            output_file=output_file,
            output_format=output_format,
            metrics=metrics,
            input_column=args.input_column,
            output_column=args.output_column
        )
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
