# Data Conversion Utility

The `prepare_dataset.py` script converts processed documents and logs into formats suitable for fine-tuning Llama models. It handles train/validation/test splitting and supports customizable prompt templates.

## Features

- Combines processed document data and log data into a unified dataset
- Supports multiple output formats (JSONL, CSV, Parquet)
- Creates train/validation/test splits with configurable ratios
- Implements customizable prompt templates for different use cases
- Balances document and log samples if needed
- Preserves metadata from source files

## Usage

```bash
python scripts/prepare_dataset.py \
    --docs_dir data/processed/documents/ \
    --logs_dir data/processed/logs/ \
    --output_dir data/processed/dataset/ \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1 \
    --format jsonl \
    --template default
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--docs_dir` | Directory containing processed documents (default: `data/processed/documents/`) |
| `--logs_dir` | Directory containing processed logs (default: `data/processed/logs/`) |
| `--output_dir` | Directory to save the dataset (default: `data/processed/dataset/`) |
| `--train_split` | Proportion of data to use for training (default: 0.8) |
| `--val_split` | Proportion of data to use for validation (default: 0.1) |
| `--test_split` | Proportion of data to use for testing (default: 0.1) |
| `--format` | Output format: jsonl, csv, or parquet (default: jsonl) |
| `--seed` | Random seed for data splitting (default: 42) |
| `--max_samples` | Maximum number of samples to include (default: all) |
| `--balance` | Balance document and log samples |
| `--template` | Prompt template to use: default, alpaca, or custom (default: default) |
| `--custom_template_file` | Path to custom template JSON file (required if template=custom) |

## Examples

### Basic Usage

Convert all documents and logs with default settings:

```bash
python scripts/prepare_dataset.py
```

### Custom Split Ratios

Specify custom train/validation/test split ratios:

```bash
python scripts/prepare_dataset.py --train_split 0.7 --val_split 0.15 --test_split 0.15
```

### Balance Document and Log Samples

Ensure equal representation of documents and logs:

```bash
python scripts/prepare_dataset.py --balance
```

### Use a Different Template

Use the Alpaca instruction format for prompts:

```bash
python scripts/prepare_dataset.py --template alpaca
```

### Using a Custom Template

Create a custom template file (`custom_template.json`):

```json
{
  "document": {
    "prompt": "Document: {title}\n\n{content}\n\nQuestion: {question}",
    "completion": "Answer: {answer}"
  },
  "log": {
    "prompt": "Log data:\n{log_content}\n\nAnalyze this log entry:",
    "completion": "{analysis}"
  }
}
```

Then use it with:

```bash
python scripts/prepare_dataset.py --template custom --custom_template_file custom_template.json
```

### Change Output Format

Save the dataset in Parquet format:

```bash
python scripts/prepare_dataset.py --format parquet
```

## Output

The script creates the following files in the output directory:

- `train.{format}`: Training dataset
- `validation.{format}`: Validation dataset
- `test.{format}`: Test dataset
- `sample.{format}`: Small sample of the dataset for inspection

Each dataset file contains entries with the following fields:

- `prompt`: The input prompt for the model
- `response`: The target completion
- `source`: Either "document" or "log"
- Additional metadata depending on the source type

## Prompt Templates

### Default Template

For documents:
```
Using the following document information, answer questions about {title}.

{content}

Question: {question}
```

For logs:
```
Analyze the following database log entry:

{log_content}

What does this log entry indicate?
```

### Alpaca Template

For documents:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question based on the document information.

### Input:
Document: {title}

{content}

Question: {question}

### Response:
```

For logs:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the database log entry and explain what it indicates.

### Input:
{log_content}

### Response:
```

## Integration with Other Scripts

This utility is typically used after processing documents and logs:

1. Process documents: `process_docs.py`
2. Process logs: `process_logs.py`
3. Prepare dataset: `prepare_dataset.py`
4. Run fine-tuning: `run_finetuning.py`

Example workflow:

```bash
# Process documents
python scripts/process_docs.py --input_dir data/raw/documents/ --output_dir data/processed/documents/

# Process logs
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/

# Prepare dataset
python scripts/prepare_dataset.py

# Run fine-tuning
python scripts/run_finetuning.py --config config/finetune_config.yaml
```
