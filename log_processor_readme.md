# Database Log Processor

This module provides enhanced processing of database logs for fine-tuning Llama 3.3 models.

## Key Features

- **Multiple Log Formats**: Processes SQL, JSON, and CSV log formats
- **Automatic Format Detection**: Detects log format based on content and file extension
- **Timestamp Extraction**: Automatically extracts and standardizes timestamps
- **Time-Based Filtering**: Filter logs based on date/time ranges
- **Query Structure Preservation**: Maintains SQL query structure and formatting
- **Metadata Extraction**: Extracts tables, schemas, and other metadata
- **Configurable Error Handling**: Options for how to handle processing errors
- **Consistent JSON Output**: Standardized output format for all log types

## Usage

### Basic Usage

```python
from src.data.db_processor import DBLogProcessor

# Initialize processor
processor = DBLogProcessor(
    input_dir="data/raw/logs/",
    output_dir="data/processed/logs/",
    format="auto",  # Auto-detect format (or specify "sql", "json", "csv")
    error_handling="skip"  # Skip files with errors
)

# Process all logs
stats = processor.process_all()
print(f"Processed {stats['total_entries']} log entries")
```

### Command Line Usage

```bash
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --format auto
```

### Time Filtering

Filter logs by date range:

```bash
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --start_date 2023-01-01 --end_date 2023-12-31
```

### Output Formats

Choose between JSON (default) or JSONL output:

```bash
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --output_format jsonl
```

### Error Handling

Configure how to handle processing errors:

```bash
# Skip files with errors (default)
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --error_handling skip

# Show warnings for errors but continue
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --error_handling warn

# Fail on first error
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --error_handling fail
```

## Log Entry Structure

All log entries are standardized to the following format:

```json
{
  "source_file": "example.sql",
  "source_format": "sql",
  "log_type": "SELECT",
  "timestamp": "2023-01-01T12:34:56",
  "raw_content": "SELECT * FROM users;",
  "processed_content": {
    "formatted_query": "SELECT *\nFROM users;",
    "tables": ["users"],
    "schema": "public"
  },
  "metadata": {
    "query_length": 19,
    "tables_referenced": 1
  },
  "processing_info": {
    "index": 0,
    "timestamp_source": "comment"
  }
}
```

## Format-Specific Processing

### SQL Logs

- Formats queries with proper indentation and capitalization
- Extracts query type (SELECT, INSERT, UPDATE, etc.)
- Identifies tables and schemas referenced
- Extracts timestamps from comments

### JSON Logs

- Handles both single JSON objects and JSONL (JSON Lines) formats
- Preserves original structure
- Extracts and standardizes timestamps from various fields
- Identifies log type/event from fields

### CSV Logs

- Automatic delimiter detection
- Converts data to a consistent format
- Handles missing values and non-standard formats
- Extracts timestamps and metadata

## Implementation Details

The processor is modular with specialized handlers for each format:

1. `DBLogProcessor` - Main processor class
2. `SQLProcessor` - Handles SQL log files
3. `JSONProcessor` - Handles JSON log files
4. `CSVProcessor` - Handles CSV log files
5. `TimeFilter` - Manages time-based filtering
6. `ErrorHandler` - Configurable error handling
7. `FormatDetector` - Detects log formats
8. `LogEntry` - Standardized data structure

## Dependencies

- `sqlparse` - For SQL parsing and formatting
- `pandas` - For CSV handling and data manipulation
- `tqdm` - For progress bars
