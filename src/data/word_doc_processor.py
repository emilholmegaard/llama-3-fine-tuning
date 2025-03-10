"""
Enhanced Word document processor for Llama 3.3 fine-tuning.

This module extracts and processes content from Word documents (.doc and .docx),
preserving folder structure for categorization, with support for extracting
tables and images, and outputting structured JSON files.
"""

import os
import logging
import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import docx
from docx.document import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.text.paragraph import Paragraph
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.exceptions import PackageNotFoundError

import mammoth
from bs4 import BeautifulSoup
from tqdm import tqdm
import olefile  # For handling .doc (legacy) files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    """Options for Word document processing."""
    
    recursive: bool = True
    preserve_structure: bool = True
    extract_images: bool = False
    extract_tables: bool = True
    extract_metadata: bool = True
    extract_headers_footers: bool = True
    min_text_length: int = 50
    max_documents: int = -1
    file_extensions: List[str] = field(default_factory=lambda: [".docx", ".doc"])
    image_format: str = "base64"  # Options: base64, filename
    output_format: str = "json"  # Options: json, jsonl, plain_text
    include_raw_html: bool = False  # Whether to include raw HTML in output
    paragraph_separator: str = "\n\n"  # Separator between paragraphs
    table_format: str = "list"  # Options: list, html, markdown


class WordDocProcessor:
    """Enhanced Word document processor for fine-tuning."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        options: Optional[ProcessingOptions] = None,
    ):
        """
        Initialize the Word document processor.

        Args:
            input_dir: Directory containing Word documents
            output_dir: Directory to save processed documents
            options: Processing options (if None, default options are used)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.options = options or ProcessingOptions()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up image directory if needed
        if self.options.extract_images and self.options.image_format == "filename":
            self.images_dir = self.output_dir / "images"
            os.makedirs(self.images_dir, exist_ok=True)

    def find_documents(self) -> List[Path]:
        """
        Find all Word documents in the input directory.

        Returns:
            List of paths to Word documents
        """
        files = []
        search_pattern = "**/*" if self.options.recursive else "*"

        for ext in self.options.file_extensions:
            pattern = search_pattern + ext
            found_files = list(self.input_dir.glob(pattern))
            files.extend(found_files)

        logger.info(f"Found {len(files)} Word documents")
        
        if self.options.max_documents > 0:
            files = files[:self.options.max_documents]
            logger.info(f"Processing first {self.options.max_documents} documents")
            
        return files

    def get_output_path(self, input_path: Path) -> Path:
        """
        Get the output path for a processed document.

        Args:
            input_path: Path to the input document

        Returns:
            Path to the output file
        """
        suffix = f".{self.options.output_format.lower()}"
        
        if self.options.preserve_structure:
            # Preserve the folder structure
            rel_path = input_path.relative_to(self.input_dir)
            output_path = self.output_dir / rel_path.with_suffix(suffix)
            os.makedirs(output_path.parent, exist_ok=True)
        else:
            # Flatten the structure
            output_path = self.output_dir / f"{input_path.stem}{suffix}"

        return output_path