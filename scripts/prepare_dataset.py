#!/usr/bin/env python
"""
Script to convert processed documents and logs to the required training format.

This script combines processed Word documents and database logs into a unified
dataset suitable for fine-tuning Llama models, with train/val/test splits.
"""

import os
import sys
import json
import logging
import argparse
import random
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_preparation.log')
    ]
)
logger = logging.getLogger('dataset_preparation')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare training dataset for fine-tuning')
    
    parser.add_argument(
        '--docs_dir',
        type=str,
        default='data/processed/documents/',
        help='Directory containing processed documents'
    )
    
    parser.add_argument(
        '--logs_dir',
        type=str,
        default='data/processed/logs/',
        help='Directory containing processed logs'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/dataset/',
        help='Directory to save the dataset'
    )
    
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Proportion of data to use for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Proportion of data to use for validation (default: 0.1)'
    )
    
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Proportion of data to use for testing (default: 0.1)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='jsonl',
        choices=['jsonl', 'csv', 'parquet'],
        help='Output format (default: jsonl)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting (default: 42)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to include (default: all)'
    )
    
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance document and log samples'
    )
    
    parser.add_argument(
        '--template',
        type=str,
        default='default',
        choices=['default', 'alpaca', 'custom'],
        help='Prompt template to use (default: default)'
    )
    
    parser.add_argument(
        '--custom_template_file',
        type=str,
        help='Path to custom template JSON file (required if template=custom)'
    )
    
    return parser.parse_args()


def load_document_data(docs_dir: str) -> List[Dict[str, Any]]:
    """
    Load processed document data.
    
    Args:
        docs_dir: Directory containing processed document files
        
    Returns:
        List of document data dictionaries
    """
    logger.info(f"Loading document data from {docs_dir}")
    documents = []
    
    # Create a Path object for the directory
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning(f"Document directory {docs_dir} not found")
        return documents
    
    # Recursively find all JSON files
    json_files = list(docs_path.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} document files")
    
    for json_file in tqdm(json_files, desc="Loading documents"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Extract relative path for categorization
            rel_path = json_file.relative_to(docs_path).parent
            category = str(rel_path) if rel_path != Path('.') else 'general'
            
            # Add category and source
            doc_data['category'] = category
            doc_data['source'] = 'document'
            doc_data['file_path'] = str(json_file)
            
            documents.append(doc_data)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents successfully")
    return documents


def load_log_data(logs_dir: str) -> List[Dict[str, Any]]:
    """
    Load processed log data.
    
    Args:
        logs_dir: Directory containing processed log files
        
    Returns:
        List of log data dictionaries
    """
    logger.info(f"Loading log data from {logs_dir}")
    logs = []
    
    # Create a Path object for the directory
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        logger.warning(f"Logs directory {logs_dir} not found")
        return logs
    
    # Recursively find all JSON files
    json_files = list(logs_path.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} log files")
    
    for json_file in tqdm(json_files, desc="Loading logs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                log_entries = json.load(f)
            
            # Extract log type from directory name
            log_type = json_file.parent.name
            
            # Process each log entry
            for entry in log_entries:
                entry['log_type'] = log_type
                entry['source'] = 'log'
                entry['file_path'] = str(json_file)
                logs.append(entry)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(logs)} log entries successfully")
    return logs


def create_prompt_completion_pairs(
    data: List[Dict[str, Any]],
    template: str = 'default',
    custom_template_file: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Create prompt-completion pairs from the data.
    
    Args:
        data: List of data entries
        template: Template to use for formatting
        custom_template_file: Path to custom template file
        
    Returns:
        List of prompt-completion pairs
    """
    logger.info(f"Creating prompt-completion pairs using {template} template")
    
    # Load templates
    templates = {
        "default": {
            "document": {
                "prompt": "Using the following document information, answer questions about {title}.\n\n{content}\n\nQuestion: {question}",
                "completion": "{answer}"
            },
            "log": {
                "prompt": "Analyze the following database log entry:\n\n{log_content}\n\nWhat does this log entry indicate?",
                "completion": "{analysis}"
            }
        },
        "alpaca": {
            "document": {
                "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the question based on the document information.\n\n### Input:\nDocument: {title}\n\n{content}\n\nQuestion: {question}\n\n### Response:",
                "completion": "{answer}"
            },
            "log": {
                "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnalyze the database log entry and explain what it indicates.\n\n### Input:\n{log_content}\n\n### Response:",
                "completion": "{analysis}"
            }
        }
    }
    
    # Load custom template if specified
    if template == 'custom' and custom_template_file:
        try:
            with open(custom_template_file, 'r', encoding='utf-8') as f:
                custom_templates = json.load(f)
            templates['custom'] = custom_templates
            template = 'custom'
        except Exception as e:
            logger.error(f"Error loading custom template: {e}")
            logger.info("Falling back to default template")
            template = 'default'
    
    pairs = []
    
    for item in tqdm(data, desc="Creating prompt-completion pairs"):
        try:
            source = item.get('source', 'document')
            
            if source == 'document':
                # For documents, create Q&A pairs
                content = item.get('content', '')
                title = item.get('title', 'Untitled Document')
                
                # If the document has predefined QA pairs, use them
                qa_pairs = item.get('qa_pairs', [])
                if not qa_pairs:
                    # Create a generic question if none exist
                    qa_pairs = [{'question': 'Summarize this document.', 'answer': f"This document is about {title}."}]
                
                for qa in qa_pairs:
                    question = qa.get('question', '')
                    answer = qa.get('answer', '')
                    
                    # Format using template
                    prompt = templates[template]['document']['prompt'].format(
                        title=title,
                        content=content,
                        question=question
                    )
                    
                    completion = templates[template]['document']['completion'].format(
                        answer=answer
                    )
                    
                    pairs.append({
                        'prompt': prompt,
                        'response': completion,
                        'source': 'document',
                        'category': item.get('category', 'general'),
                        'file_path': item.get('file_path', '')
                    })
            
            elif source == 'log':
                # For logs, create analysis prompts
                log_content = item.get('content', '')
                analysis = item.get('analysis', 'No analysis available.')
                
                # Format using template
                prompt = templates[template]['log']['prompt'].format(
                    log_content=log_content
                )
                
                completion = templates[template]['log']['completion'].format(
                    analysis=analysis
                )
                
                pairs.append({
                    'prompt': prompt,
                    'response': completion,
                    'source': 'log',
                    'log_type': item.get('log_type', 'general'),
                    'file_path': item.get('file_path', '')
                })
        
        except Exception as e:
            logger.error(f"Error creating pair for item: {e}")
    
    logger.info(f"Created {len(pairs)} prompt-completion pairs")
    return pairs


def split_data(
    data: List[Dict[str, str]],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: List of data entries
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Normalize split proportions
    total = train_split + val_split + test_split
    train_split /= total
    val_split /= total
    test_split /= total
    
    logger.info(f"Splitting data: {train_split:.1%} train, {val_split:.1%} validation, {test_split:.1%} test")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    n = len(shuffled_data)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    # Split data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    logger.info(f"Split sizes: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    return train_data, val_data, test_data


def balance_data(docs: List[Dict[str, Any]], logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Balance document and log data.
    
    Args:
        docs: Document data
        logs: Log data
        
    Returns:
        Combined and balanced data
    """
    logger.info(f"Balancing data: {len(docs)} documents, {len(logs)} logs")
    
    # Find the smaller dataset size
    min_size = min(len(docs), len(logs))
    
    # Sample from the larger dataset
    if len(docs) > min_size:
        docs = random.sample(docs, min_size)
    if len(logs) > min_size:
        logs = random.sample(logs, min_size)
    
    # Combine data
    combined = docs + logs
    random.shuffle(combined)
    
    logger.info(f"Balanced data: {len(combined)} total samples ({len(docs)} docs, {len(logs)} logs)")
    return combined


def save_data(data: List[Dict[str, str]], output_path: str, format: str = 'jsonl'):
    """
    Save data to file in the specified format.
    
    Args:
        data: Data to save
        output_path: Path to save the data
        format: Output format (jsonl, csv, parquet)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving {len(data)} samples to {output_path}")
    
    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    elif format == 'csv':
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    elif format == 'parquet':
        # Convert to DataFrame and save as Parquet
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
    
    logger.info(f"Data saved successfully to {output_path}")


def main():
    """Main entry point for dataset preparation."""
    # Parse arguments
    args = parse_args()
    
    # Validate splits sum to 1
    total_split = args.train_split + args.val_split + args.test_split
    if not 0.99 <= total_split <= 1.01:  # Allow for small floating point errors
        logger.warning(f"Split proportions sum to {total_split}, not 1.0. Normalizing...")
    
    # Load document data
    documents = load_document_data(args.docs_dir)
    
    # Load log data
    logs = load_log_data(args.logs_dir)
    
    # Balance data if requested
    if args.balance:
        combined_data = balance_data(documents, logs)
    else:
        combined_data = documents + logs
        random.shuffle(combined_data)
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(combined_data):
        logger.info(f"Limiting to {args.max_samples} samples")
        combined_data = combined_data[:args.max_samples]
    
    # Create prompt-completion pairs
    pairs = create_prompt_completion_pairs(
        combined_data,
        template=args.template,
        custom_template_file=args.custom_template_file
    )
    
    # Split data
    train_data, val_data, test_data = split_data(
        pairs,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets
    save_data(
        train_data,
        os.path.join(args.output_dir, f"train.{args.format}"),
        args.format
    )
    
    save_data(
        val_data,
        os.path.join(args.output_dir, f"validation.{args.format}"),
        args.format
    )
    
    save_data(
        test_data,
        os.path.join(args.output_dir, f"test.{args.format}"),
        args.format
    )
    
    # Save a sample of the data for inspection
    sample_size = min(5, len(pairs))
    sample_data = random.sample(pairs, sample_size)
    save_data(
        sample_data,
        os.path.join(args.output_dir, f"sample.{args.format}"),
        args.format
    )
    
    logger.info("Dataset preparation completed successfully")


if __name__ == "__main__":
    main()
