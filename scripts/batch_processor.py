#!/usr/bin/env python
"""
Batch Processor for Document and Log Data.

This script provides a unified system for processing batches of documents and logs,
with configurable pipelines, parallel processing, and progress tracking.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import pandas as pd
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import shutil
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.word_processor import process_word_document
from src.data.db_processor import process_log_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_processing.log')
    ]
)
logger = logging.getLogger('batch_processor')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch Processor for Documents and Logs')
    
    # General arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Input/output arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--input_dirs',
        type=str,
        nargs='+',
        help='Directories containing input files'
    )
    io_group.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save processed files'
    )
    io_group.add_argument(
        '--file_types',
        type=str,
        nargs='+',
        default=['docx', 'doc', 'log', 'sql', 'json', 'csv'],
        help='File types to process'
    )
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively process subdirectories'
    )
    proc_group.add_argument(
        '--preserve_structure',
        action='store_true',
        help='Preserve directory structure in output'
    )
    proc_group.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker processes'
    )
    proc_group.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        help='Chunk size for text processing'
    )
    proc_group.add_argument(
        '--clean_existing',
        action='store_true',
        help='Clean existing output directories'
    )
    
    # Document-specific options
    doc_group = parser.add_argument_group('Document Options')
    doc_group.add_argument(
        '--extract_images',
        action='store_true',
        help='Extract images from documents'
    )
    doc_group.add_argument(
        '--extract_tables',
        action='store_true',
        help='Extract tables from documents'
    )
    doc_group.add_argument(
        '--extract_headers',
        action='store_true',
        help='Extract headers from documents'
    )
    
    # Log-specific options
    log_group = parser.add_argument_group('Log Options')
    log_group.add_argument(
        '--log_format',
        type=str,
        default='auto',
        choices=['auto', 'sql', 'json', 'csv', 'text'],
        help='Format of log files'
    )
    log_group.add_argument(
        '--parse_timestamps',
        action='store_true',
        help='Parse timestamps in logs'
    )
    log_group.add_argument(
        '--extract_queries',
        action='store_true',
        help='Extract SQL queries from logs'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    # Check if file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file type
    ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if ext in ['.json']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        elif ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                logger.error("YAML library not installed. Install with 'pip install pyyaml'")
                raise
        
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        logger.info("Configuration loaded successfully")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def collect_files(
    input_dirs: List[str],
    file_types: List[str],
    recursive: bool = False
) -> Dict[str, List[str]]:
    """
    Collect files of specified types from input directories.
    
    Args:
        input_dirs: List of input directories
        file_types: List of file extensions to include
        recursive: Whether to search subdirectories
        
    Returns:
        Dictionary mapping file types to lists of file paths
    """
    logger.info(f"Collecting files from {len(input_dirs)} directories")
    
    file_types = [f.lower().lstrip('.') for f in file_types]
    file_map = {ft: [] for ft in file_types}
    
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            logger.warning(f"Input directory not found: {input_dir}")
            continue
        
        logger.info(f"Searching in {input_dir}")
        
        for file_type in file_types:
            # Create glob pattern
            pattern = f"**/*.{file_type}" if recursive else f"*.{file_type}"
            
            # Find matching files
            matching_files = glob.glob(os.path.join(input_dir, pattern), recursive=recursive)
            file_map[file_type].extend(matching_files)
            
            logger.info(f"Found {len(matching_files)} {file_type} files in {input_dir}")
    
    # Log summary
    total_files = sum(len(files) for files in file_map.values())
    logger.info(f"Collected {total_files} files in total")
    
    return file_map


def process_document(
    file_path: str,
    output_dir: str,
    preserve_structure: bool = False,
    extract_images: bool = False,
    extract_tables: bool = False,
    extract_headers: bool = False,
    chunk_size: int = 1000
) -> Dict[str, Any]:
    """
    Process a Word document.
    
    Args:
        file_path: Path to the document
        output_dir: Directory to save processed output
        preserve_structure: Whether to preserve directory structure
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        extract_headers: Whether to extract headers
        chunk_size: Chunk size for text processing
        
    Returns:
        Processing result dictionary
    """
    try:
        start_time = time.time()
        
        # Determine output path
        rel_path = ""
        if preserve_structure:
            # Get relative path from input directory to file
            file_dir = os.path.dirname(file_path)
            rel_path = os.path.relpath(file_dir, os.path.dirname(os.path.dirname(file_path)))
            if rel_path == '.':
                rel_path = ''
        
        output_subdir = os.path.join(output_dir, "documents", rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process document
        result = process_word_document(
            file_path,
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_headers=extract_headers,
            chunk_size=chunk_size
        )
        
        # Save result
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_subdir, f"{file_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save images if extracted
        if extract_images and 'images' in result and result['images']:
            images_dir = os.path.join(output_subdir, f"{file_name}_images")
            os.makedirs(images_dir, exist_ok=True)
            
            for i, img_data in enumerate(result['images']):
                img_path = os.path.join(images_dir, f"image_{i+1}.png")
                
                # Save the image if it contains binary data
                if 'data' in img_data and img_data['data']:
                    try:
                        import base64
                        with open(img_path, 'wb') as img_file:
                            img_file.write(base64.b64decode(img_data['data']))
                        
                        # Remove binary data from JSON result to save space
                        result['images'][i]['file_path'] = os.path.relpath(img_path, output_subdir)
                        del result['images'][i]['data']
                    except Exception as e:
                        logger.warning(f"Error saving image from {file_path}: {str(e)}")
            
            # Update the saved result without image binary data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'file_path': file_path,
            'output_path': output_path,
            'success': True,
            'processing_time': processing_time,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"Error processing document {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        return {
            'file_path': file_path,
            'output_path': None,
            'success': False,
            'processing_time': 0,
            'error': str(e)
        }


def process_log(
    file_path: str,
    output_dir: str,
    preserve_structure: bool = False,
    log_format: str = 'auto',
    parse_timestamps: bool = False,
    extract_queries: bool = False
) -> Dict[str, Any]:
    """
    Process a log file.
    
    Args:
        file_path: Path to the log file
        output_dir: Directory to save processed output
        preserve_structure: Whether to preserve directory structure
        log_format: Format of the log file
        parse_timestamps: Whether to parse timestamps
        extract_queries: Whether to extract SQL queries
        
    Returns:
        Processing result dictionary
    """
    try:
        start_time = time.time()
        
        # Determine output path
        rel_path = ""
        if preserve_structure:
            # Get relative path from input directory to file
            file_dir = os.path.dirname(file_path)
            rel_path = os.path.relpath(file_dir, os.path.dirname(os.path.dirname(file_path)))
            if rel_path == '.':
                rel_path = ''
        
        output_subdir = os.path.join(output_dir, "logs", rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Determine format if auto
        if log_format == 'auto':
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext in ['sql']:
                log_format = 'sql'
            elif ext in ['json']:
                log_format = 'json'
            elif ext in ['csv']:
                log_format = 'csv'
            else:
                log_format = 'text'
        
        # Process log
        result = process_log_file(
            file_path,
            format_type=log_format,
            parse_timestamps=parse_timestamps,
            extract_queries=extract_queries
        )
        
        # Save result
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_subdir, f"{file_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'file_path': file_path,
            'output_path': output_path,
            'success': True,
            'processing_time': processing_time,
            'error': None
        }
    
    except Exception as e:
        error_msg = f"Error processing log {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        return {
            'file_path': file_path,
            'output_path': None,
            'success': False,
            'processing_time': 0,
            'error': str(e)
        }


def clean_output_directory(output_dir: str):
    """
    Clean the output directory.
    
    Args:
        output_dir: Directory to clean
    """
    logger.info(f"Cleaning output directory: {output_dir}")
    
    subdirs = ["documents", "logs"]
    
    for subdir in subdirs:
        full_path = os.path.join(output_dir, subdir)
        if os.path.exists(full_path):
            try:
                shutil.rmtree(full_path)
                logger.info(f"Removed directory: {full_path}")
            except Exception as e:
                logger.error(f"Error removing directory {full_path}: {str(e)}")
        
        # Recreate empty directory
        os.makedirs(full_path, exist_ok=True)


def batch_process_files(
    file_map: Dict[str, List[str]],
    output_dir: str,
    workers: int = 4,
    preserve_structure: bool = False,
    # Document options
    extract_images: bool = False,
    extract_tables: bool = False,
    extract_headers: bool = False,
    chunk_size: int = 1000,
    # Log options
    log_format: str = 'auto',
    parse_timestamps: bool = False,
    extract_queries: bool = False
) -> Dict[str, Any]:
    """
    Process multiple files in parallel.
    
    Args:
        file_map: Dictionary mapping file types to file paths
        output_dir: Directory to save processed output
        workers: Number of worker processes
        preserve_structure: Whether to preserve directory structure
        extract_images: Whether to extract images from documents
        extract_tables: Whether to extract tables from documents
        extract_headers: Whether to extract headers from documents
        chunk_size: Chunk size for text processing
        log_format: Format of log files
        parse_timestamps: Whether to parse timestamps in logs
        extract_queries: Whether to extract SQL queries from logs
        
    Returns:
        Processing statistics
    """
    logger.info(f"Starting batch processing with {workers} workers")
    
    # Prepare output directories
    os.makedirs(os.path.join(output_dir, "documents"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Collect all files
    document_files = []
    log_files = []
    
    # Categorize files
    for file_type, files in file_map.items():
        if file_type in ['doc', 'docx']:
            document_files.extend(files)
        elif file_type in ['log', 'sql', 'json', 'csv']:
            log_files.extend(files)
    
    logger.info(f"Processing {len(document_files)} documents and {len(log_files)} logs")
    
    # Initialize results
    results = {
        'total_files': len(document_files) + len(log_files),
        'successful': 0,
        'failed': 0,
        'processing_time': 0,
        'document_count': len(document_files),
        'log_count': len(log_files),
        'errors': []
    }
    
    # Process documents
    if document_files:
        logger.info(f"Processing {len(document_files)} documents")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit document processing tasks
            future_to_file = {
                executor.submit(
                    process_document,
                    file_path=file_path,
                    output_dir=output_dir,
                    preserve_structure=preserve_structure,
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    extract_headers=extract_headers,
                    chunk_size=chunk_size
                ): file_path for file_path in document_files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Documents"):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'file_path': file_path,
                            'error': result['error']
                        })
                    
                    results['processing_time'] += result['processing_time']
                
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append({
                        'file_path': file_path,
                        'error': str(e)
                    })
    
    # Process logs
    if log_files:
        logger.info(f"Processing {len(log_files)} log files")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit log processing tasks
            future_to_file = {
                executor.submit(
                    process_log,
                    file_path=file_path,
                    output_dir=output_dir,
                    preserve_structure=preserve_structure,
                    log_format=log_format,
                    parse_timestamps=parse_timestamps,
                    extract_queries=extract_queries
                ): file_path for file_path in log_files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Logs"):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'file_path': file_path,
                            'error': result['error']
                        })
                    
                    results['processing_time'] += result['processing_time']
                
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append({
                        'file_path': file_path,
                        'error': str(e)
                    })
    
    # Calculate success rate
    if results['total_files'] > 0:
        results['success_rate'] = results['successful'] / results['total_files']
    else:
        results['success_rate'] = 0
    
    logger.info(f"Batch processing completed: {results['successful']} succeeded, {results['failed']} failed")
    
    return results


def save_processing_report(results: Dict[str, Any], output_dir: str):
    """
    Save a processing report.
    
    Args:
        results: Processing results dictionary
        output_dir: Directory to save the report
    """
    logger.info("Generating processing report")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"processing_report_{timestamp}.json")
    
    # Add timestamp to results
    results['timestamp'] = timestamp
    
    # Save JSON report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Also create a markdown summary
    md_report_path = os.path.join(output_dir, f"processing_report_{timestamp}.md")
    
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write("# Batch Processing Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Files:** {results['total_files']}\n")
        f.write(f"- **Successful:** {results['successful']}\n")
        f.write(f"- **Failed:** {results['failed']}\n")
        f.write(f"- **Success Rate:** {results['success_rate']*100:.2f}%\n")
        f.write(f"- **Total Processing Time:** {results['processing_time']:.2f} seconds\n\n")
        
        f.write("## File Counts\n\n")
        f.write(f"- **Documents:** {results['document_count']}\n")
        f.write(f"- **Logs:** {results['log_count']}\n\n")
        
        if results['errors']:
            f.write("## Errors\n\n")
            f.write("| File | Error |\n")
            f.write("|------|-------|\n")
            
            for error in results['errors']:
                file_name = os.path.basename(error['file_path'])
                f.write(f"| {file_name} | {error['error']} |\n")
    
    logger.info(f"Report saved to {report_path} and {md_report_path}")


def main():
    """Main entry point for batch processor."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            config = load_config(args.config)
            
            # Override command-line args with config values
            if 'input_dirs' in config:
                args.input_dirs = config['input_dirs']
            if 'output_dir' in config:
                args.output_dir = config['output_dir']
            if 'file_types' in config:
                args.file_types = config['file_types']
            
            # Set other options from config
            for key, value in config.items():
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, value)
        
        # Validate input directories
        if not args.input_dirs:
            logger.error("No input directories specified")
            sys.exit(1)
        
        # Clean output directory if requested
        if args.clean_existing:
            clean_output_directory(args.output_dir)
        
        # Collect files to process
        file_map = collect_files(
            input_dirs=args.input_dirs,
            file_types=args.file_types,
            recursive=args.recursive
        )
        
        # Check if any files were found
        total_files = sum(len(files) for files in file_map.values())
        if total_files == 0:
            logger.warning("No files found matching the specified criteria")
            sys.exit(0)
        
        # Process files
        results = batch_process_files(
            file_map=file_map,
            output_dir=args.output_dir,
            workers=args.workers,
            preserve_structure=args.preserve_structure,
            # Document options
            extract_images=args.extract_images,
            extract_tables=args.extract_tables,
            extract_headers=args.extract_headers,
            chunk_size=args.chunk_size,
            # Log options
            log_format=args.log_format,
            parse_timestamps=args.parse_timestamps,
            extract_queries=args.extract_queries
        )
        
        # Save processing report
        save_processing_report(results, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print(f"Processing completed: {results['successful']}/{results['total_files']} files processed successfully")
        print(f"Success rate: {results['success_rate']*100:.2f}%")
        print(f"Total processing time: {results['processing_time']:.2f} seconds")
        print("="*50 + "\n")
        
        # Exit with error code if any files failed
        if results['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
