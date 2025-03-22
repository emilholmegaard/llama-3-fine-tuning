# Batch Processor

The `batch_processor.py` script provides a unified system for processing batches of documents and logs with parallel processing, configurable pipelines, and detailed reporting.

## Features

- Process Word documents and log files in parallel
- Preserve directory structures during processing
- Extract images, tables, and headers from documents
- Parse timestamps and extract queries from logs
- Generate detailed processing reports
- Support for configuration files

## Usage

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/documents/ data/raw/logs/ \
    --output_dir data/processed/ \
    --recursive \
    --preserve_structure \
    --workers 4 \
    --extract_images \
    --extract_tables
```

## Arguments

### General Options

| Argument | Description |
|----------|-------------|
| `--config` | Path to configuration file |

### Input/Output Options

| Argument | Description |
|----------|-------------|
| `--input_dirs` | Directories containing input files |
| `--output_dir` | Directory to save processed files (default: data/processed) |
| `--file_types` | File types to process (default: docx, doc, log, sql, json, csv) |

### Processing Options

| Argument | Description |
|----------|-------------|
| `--recursive` | Recursively process subdirectories |
| `--preserve_structure` | Preserve directory structure in output |
| `--workers` | Number of worker processes (default: 4) |
| `--chunk_size` | Chunk size for text processing (default: 1000) |
| `--clean_existing` | Clean existing output directories |

### Document Options

| Argument | Description |
|----------|-------------|
| `--extract_images` | Extract images from documents |
| `--extract_tables` | Extract tables from documents |
| `--extract_headers` | Extract headers from documents |

### Log Options

| Argument | Description |
|----------|-------------|
| `--log_format` | Format of log files: auto, sql, json, csv, text (default: auto) |
| `--parse_timestamps` | Parse timestamps in logs |
| `--extract_queries` | Extract SQL queries from logs |

## Examples

### Basic Usage

Process all supported files in the input directories:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/documents/ data/raw/logs/
```

### Recursive Processing with Structure Preservation

Process all files recursively, preserving directory structure:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/ \
    --recursive \
    --preserve_structure
```

### Document Processing with Image and Table Extraction

Process documents and extract images and tables:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/documents/ \
    --extract_images \
    --extract_tables \
    --extract_headers
```

### Log Processing with Special Options

Process log files with timestamp parsing and query extraction:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/logs/ \
    --log_format sql \
    --parse_timestamps \
    --extract_queries
```

### Parallel Processing with Custom Worker Count

Process files using 8 parallel workers:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/documents/ data/raw/logs/ \
    --workers 8
```

### Using a Configuration File

Create a JSON or YAML configuration file (`batch_config.json`):

```json
{
  "input_dirs": ["data/raw/documents/", "data/raw/logs/"],
  "output_dir": "data/processed/",
  "recursive": true,
  "preserve_structure": true,
  "workers": 8,
  "extract_images": true,
  "extract_tables": true,
  "parse_timestamps": true,
  "extract_queries": true
}
```

Then use it with:

```bash
python scripts/batch_processor.py --config batch_config.json
```

### Clean Existing Outputs

Clean output directories before processing:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/ \
    --clean_existing
```

## Output Structure

The batch processor organizes processed files into the following structure:

```
output_dir/
├── documents/
│   ├── document1.json
│   ├── document1_images/
│   │   ├── image_1.png
│   │   └── ...
│   ├── subdirectory/ (if preserve_structure is enabled)
│   │   ├── document2.json
│   │   └── ...
│   └── ...
├── logs/
│   ├── log1.json
│   ├── subdirectory/ (if preserve_structure is enabled)
│   │   ├── log2.json
│   │   └── ...
│   └── ...
├── processing_report_20250322_083045.json
└── processing_report_20250322_083045.md
```

## Processed Document Format

Documents are processed and saved as JSON files with the following structure:

```json
{
  "title": "Document Title",
  "content": "Full document text content...",
  "metadata": {
    "author": "Author Name",
    "created_date": "2025-01-15T10:30:00",
    "modified_date": "2025-02-20T14:45:00",
    "page_count": 5
  },
  "sections": [
    {
      "heading": "Section 1",
      "level": 1,
      "content": "Section 1 content..."
    },
    ...
  ],
  "tables": [
    {
      "caption": "Table 1",
      "data": [["Header1", "Header2"], ["Value1", "Value2"], ...]
    },
    ...
  ],
  "images": [
    {
      "caption": "Image 1",
      "description": "Description of image 1",
      "file_path": "document1_images/image_1.png"
    },
    ...
  ]
}
```

## Processed Log Format

Logs are processed and saved as JSON files with the following structure:

```json
[
  {
    "timestamp": "2025-03-15T14:30:45.123",
    "level": "ERROR",
    "message": "Database connection timeout",
    "content": "Full log entry content...",
    "query": "SELECT * FROM users WHERE id = 123",
    "context": {
      "connection_id": "conn-1234",
      "user": "system",
      "duration_ms": 5000
    }
  },
  ...
]
```

## Processing Report

After processing, a detailed report is generated in both JSON and Markdown formats:

### JSON Report

```json
{
  "timestamp": "20250322_083045",
  "total_files": 150,
  "successful": 145,
  "failed": 5,
  "processing_time": 25.45,
  "document_count": 100,
  "log_count": 50,
  "success_rate": 0.9667,
  "errors": [
    {
      "file_path": "data/raw/documents/corrupted.docx",
      "error": "Error processing document: File is corrupted"
    },
    ...
  ]
}
```

### Markdown Report

```markdown
# Batch Processing Report

**Generated:** 2025-03-22 08:30:45

## Summary

- **Total Files:** 150
- **Successful:** 145
- **Failed:** 5
- **Success Rate:** 96.67%
- **Total Processing Time:** 25.45 seconds

## File Counts

- **Documents:** 100
- **Logs:** 50

## Errors

| File | Error |
|------|-------|
| corrupted.docx | Error processing document: File is corrupted |
| ... | ... |
```

## Integration with Other Scripts

This utility is typically the first step in the fine-tuning pipeline:

1. Process data: `batch_processor.py`
2. Prepare dataset: `prepare_dataset.py`
3. Fine-tune model: `run_finetuning.py`

Example workflow:

```bash
# Process raw data
python scripts/batch_processor.py \
    --input_dirs data/raw/ \
    --recursive \
    --extract_images \
    --extract_tables

# Prepare dataset
python scripts/prepare_dataset.py

# Run fine-tuning
python scripts/run_finetuning.py --config config/finetune_config.yaml
```

## Tips for Effective Use

1. **Workers**: Set the number of workers based on your CPU cores. Too many workers might lead to memory issues for large documents.

2. **Chunk Size**: Adjust chunk size for text processing based on your document complexity. Smaller chunks may be better for highly technical content.

3. **Configuration File**: For complex processing needs, use a configuration file instead of command-line arguments to make your workflow more reproducible.

4. **Preserve Structure**: When working with organized document repositories, use the `--preserve_structure` flag to maintain the same organization in processed outputs.

5. **Selective Processing**: Process only what you need. For example, if you don't need images, avoid using `--extract_images` to speed up processing.
