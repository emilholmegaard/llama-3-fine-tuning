#!/usr/bin/env python
"""
Script for processing Word documents for Llama 3.3 fine-tuning.

This script provides a command-line interface for extracting and processing
content from Word documents, including text, tables, and images, while
preserving folder structure if needed.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.word_doc_processor import WordDocProcessor, ProcessingOptions


def main():
    """Main function for processing Word documents."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process Word documents for Llama fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, 
                        help="Directory containing Word documents")
    parser.add_argument("--output_dir", required=True, 
                        help="Directory to save processed documents")
    
    # Process control arguments
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--recursive", action="store_true", default=True,
                        help="Process documents in subdirectories")
    process_group.add_argument("--preserve_structure", action="store_true", default=True,
                        help="Maintain folder structure in output")
    process_group.add_argument("--min_text_length", type=int, default=50,
                        help="Minimum text length to keep")
    process_group.add_argument("--max_documents", type=int, default=-1,
                        help="Maximum number of documents to process (-1 for all)")
    
    # Content extraction arguments
    extract_group = parser.add_argument_group("Content Extraction")
    extract_group.add_argument("--extract_images", action="store_true",
                        help="Extract image data from documents")
    extract_group.add_argument("--extract_tables", action="store_true", default=True,
                        help="Extract tables from documents")
    extract_group.add_argument("--extract_metadata", action="store_true", default=True,
                        help="Extract document metadata")
    extract_group.add_argument("--extract_headers_footers", action="store_true",
                        help="Extract headers and footers from documents")
    
    # File options
    file_group = parser.add_argument_group("File Options")
    file_group.add_argument("--file_extensions", nargs="+", default=[".docx", ".doc"],
                        help="File extensions to process (e.g., .docx .doc)")
    
    # Output format arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output_format", choices=["json", "jsonl", "plain_text"],
                        default="json", help="Output format for processed documents")
    output_group.add_argument("--image_format", choices=["base64", "filename"],
                        default="base64", help="Format for extracted images")
    output_group.add_argument("--include_raw_html", action="store_true",
                        help="Include raw HTML in output (when using mammoth)")
    
    # Logging options
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set the logging level")
    log_group.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output (equivalent to --log_level DEBUG)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = args.log_level
    if args.verbose:
        log_level = "DEBUG"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create input and output directories if they don't exist
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create processing options
    options = ProcessingOptions(
        recursive=args.recursive,
        preserve_structure=args.preserve_structure,
        extract_images=args.extract_images,
        extract_tables=args.extract_tables,
        extract_metadata=args.extract_metadata,
        extract_headers_footers=args.extract_headers_footers,
        min_text_length=args.min_text_length,
        max_documents=args.max_documents,
        file_extensions=args.file_extensions,
        image_format=args.image_format,
        output_format=args.output_format,
        include_raw_html=args.include_raw_html,
    )
    
    # Create processor and process documents
    processor = WordDocProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        options=options,
    )
    
    total, succeeded = processor.process_all()
    
    # Print summary
    print(f"\nProcessing summary:")
    print(f"  Total documents: {total}")
    print(f"  Successfully processed: {succeeded}")
    print(f"  Failed: {total - succeeded}")
    print(f"  Success rate: {succeeded / total * 100:.1f}%" if total > 0 else "  Success rate: N/A")
    
    # Return status code
    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
