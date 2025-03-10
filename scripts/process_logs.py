#!/usr/bin/env python
"""
Database log processing script for Llama 3.3 fine-tuning.

This script processes various database log formats (SQL, JSON, CSV) for use in fine-tuning.
It uses the DBLogProcessor module to handle the processing.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.db_processor import DBLogProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("process_logs")


def main():
    """Main function to process database logs."""
    parser = argparse.ArgumentParser(description="Process database logs for Llama fine-tuning")
    
    # Input/output options
    parser.add_argument("--input_dir", required=True, help="Directory containing log files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed logs")
    
    # Format options
    parser.add_argument("--format", default="auto", choices=["auto", "sql", "json", "csv"], 
                        help="Log format (auto, sql, json, csv)")
    parser.add_argument("--output_format", default="json", choices=["json", "jsonl"],
                        help="Output format (json, jsonl)")
    parser.add_argument("--preserve_structure", action="store_true",
                        help="Preserve log structure in output")
    
    # Time filtering
    parser.add_argument("--start_date", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="End date filter (YYYY-MM-DD)")
    
    # Error handling
    parser.add_argument("--error_handling", default="skip", choices=["skip", "warn", "fail"],
                        help="How to handle errors (skip, warn, fail)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create input/output directories if they don't exist
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up time filter
    time_filter = {
        "start_date": args.start_date,
        "end_date": args.end_date
    }
    
    logger.info(f"Processing logs from {input_dir} to {output_dir}")
    logger.info(f"Format: {args.format}, Output format: {args.output_format}")
    
    if args.start_date or args.end_date:
        logger.info(f"Time filter: {args.start_date} to {args.end_date}")
    
    # Create and run the processor
    processor = DBLogProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format=args.format,
        time_filter=time_filter,
        error_handling=args.error_handling,
        output_format=args.output_format,
        preserve_structure=args.preserve_structure
    )
    
    # Process all logs
    stats = processor.process_all()
    
    # Print summary
    logger.info("Processing complete!")
    logger.info(f"Processed {stats['processed_files']} files with {stats['total_entries']} log entries")
    logger.info(f"Entries by format: SQL: {stats['entries_by_format'].get('sql', 0)}, "
                f"JSON: {stats['entries_by_format'].get('json', 0)}, "
                f"CSV: {stats['entries_by_format'].get('csv', 0)}")
    
    if stats['errors'] > 0:
        logger.warning(f"Encountered {stats['errors']} errors during processing")
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Time taken: {stats.get('duration_seconds', 0):.2f} seconds")


if __name__ == "__main__":
    main()
