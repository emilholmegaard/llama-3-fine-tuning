# Data Preparation

This guide explains how to prepare and process data for fine-tuning Llama 3.3 models, covering both document processing and database log integration.

## Overview

High-quality data is crucial for effective fine-tuning. This repository provides tools for processing:

1. **Word documents**: Technical documentation, user manuals, etc.
2. **Database logs**: SQL queries, transaction logs, etc.

The data preparation workflow consists of:

```
Raw Data → Preprocessing → Formatting → Dataset Creation → Fine-Tuning Input
```

## Directory Structure

Create the following directory structure for your data:

```
data/
├── raw/
│   ├── documents/      # Place raw Word documents here
│   └── logs/           # Place raw database logs here
├── processed/
│   ├── documents/      # Processed document data
│   ├── logs/           # Processed log data
│   └── dataset/        # Combined training datasets
└── models/             # Fine-tuned model outputs
```

## Processing Word Documents

### Supported Formats

- `.docx`: Microsoft Word documents
- `.doc`: Legacy Word documents (limited support)
- `.rtf`: Rich Text Format

### Processing Steps

1. **Organize your documents**:
   - Place all documents in `data/raw/documents/`
   - Consider using subdirectories to maintain categorization

2. **Run the document processing script**:

   ```bash
   python scripts/process_docs.py \
     --input_dir data/raw/documents/ \
     --output_dir data/processed/documents/ \
     --recursive \
     --preserve_structure
   ```

   Options:
   - `--input_dir`: Directory containing Word documents
   - `--output_dir`: Directory to save processed documents
   - `--recursive`: Process documents in subdirectories
   - `--preserve_structure`: Maintain folder structure in output
   - `--format`: Output format (default: jsonl)

3. **Review the processed output**:
   - Check `data/processed/documents/` for the processed files
   - Verify document structure and content preservation

### Best Practices

- **Maintain structure**: Folder hierarchy can be used as categories for training
- **Clean formatting**: Remove headers, footers, and irrelevant content
- **Consistent naming**: Use descriptive filenames
- **Include metadata**: Capture document origin, date, and category

## Processing Database Logs

### Supported Log Types

- SQL query logs
- Transaction logs
- Database operation logs 
- CSV export of database records

### Processing Steps

1. **Prepare your log files**:
   - Place all log files in `data/raw/logs/`
   - Ensure consistent formatting within each log type

2. **Run the log processing script**:

   ```bash
   python scripts/process_logs.py \
     --input_dir data/raw/logs/ \
     --output_dir data/processed/logs/ \
     --format sql
   ```

   Options:
   - `--input_dir`: Directory containing log files
   - `--output_dir`: Directory to save processed logs
   - `--format`: Log format (sql, json, csv)
   - `--filter`: Optional regex pattern to filter specific log entries
   - `--max_entries`: Maximum number of entries to process

3. **Review the processed output**:
   - Check `data/processed/logs/` for the processed files
   - Verify log parsing and structure

### Best Practices

- **Filter irrelevant entries**: Remove debugging, error logs, or monitoring entries
- **Anonymize sensitive data**: Replace personal information, credentials, or sensitive business data
- **Group related queries**: Organize logs by functionality or application area
- **Capture context**: Include session information, timestamps, or request context when relevant

## Creating Training Datasets

After processing documents and logs, combine them into training datasets:

```bash
python scripts/prepare_dataset.py \
  --docs_dir data/processed/documents/ \
  --logs_dir data/processed/logs/ \
  --output_dir data/processed/dataset/ \
  --train_split 0.8
```

Options:
- `--docs_dir`: Directory with processed documents
- `--logs_dir`: Directory with processed logs
- `--output_dir`: Directory to save the dataset
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--format`: Output format (jsonl, csv, parquet)
- `--seed`: Random seed for reproducible dataset splits

## Dataset Format

The final dataset follows this format:

```json
{
  "text": "The document or log content goes here...",
  "metadata": {
    "source": "document",
    "category": "user_manual",
    "file": "original_filename.docx",
    "path": "original/path/structure"
  }
}
```

For instruction/response formatting:

```json
{
  "instruction": "Find the table transaction with ID #12345",
  "input": "",
  "output": "SELECT * FROM transactions WHERE transaction_id = 12345;",
  "metadata": {
    "source": "log",
    "category": "query",
    "file": "query_logs_2023.txt"
  }
}
```

## Validating Your Dataset

Before fine-tuning, validate your dataset:

```bash
python scripts/validate_dataset.py --input_file data/processed/dataset/train.jsonl
```

This will:
- Check file integrity and format
- Verify all required fields are present
- Report statistics (token counts, sample distribution)
- Flag potential issues (too short/long samples, duplicates)

## Next Steps

Once your data is prepared, proceed to:
- [Basic Fine-Tuning Guide](./fine_tuning/basic_fine_tuning.md)
- [Advanced Configuration Options](./fine_tuning/advanced_configuration.md)