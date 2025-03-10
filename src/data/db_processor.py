"""
Database log processor for Llama 3.3 fine-tuning.

This module provides enhanced processing of database logs for fine-tuning Llama 3.3 models.
Key features:
- Support for SQL, JSON, and CSV log formats
- Timestamp extraction and time-based filtering
- Query structure preservation and metadata extraction
- Configurable error handling
- Consistent JSON output format
"""

import os
import logging
import json
import csv
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict

import sqlparse
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """
    Represents a processed log entry.
    
    This dataclass provides a standardized structure for all log types,
    with common fields and type-specific content.
    """
    # Common fields
    source_file: str
    source_format: str
    log_type: str
    timestamp: Optional[str] = None
    
    # Content fields
    raw_content: str = ""
    processed_content: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing info
    processing_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the log entry to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert the log entry to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


class TimeFilter:
    """
    Handles time-based filtering of log entries.
    """
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        datetime_formats: List[str] = None,
    ):
        """
        Initialize the time filter.
        
        Args:
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            datetime_formats: List of datetime formats to try when parsing timestamps
        """
        self.start_date = None
        self.end_date = None
        
        # Set default datetime formats if none provided
        self.datetime_formats = datetime_formats or [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]
        
        # Convert date strings to datetime objects
        if start_date:
            self.start_date = datetime.datetime.fromisoformat(start_date)
        if end_date:
            self.end_date = datetime.datetime.fromisoformat(end_date)
            # Set time to end of day
            self.end_date = self.end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime.datetime]:
        """
        Parse a timestamp string into a datetime object.
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not timestamp_str or not isinstance(timestamp_str, str):
            return None
            
        # Try each format
        for fmt in self.datetime_formats:
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
        
        # Try Unix timestamp (assuming seconds)
        try:
            if timestamp_str.isdigit() or (timestamp_str.replace('.', '', 1).isdigit() and timestamp_str.count('.') == 1):
                ts = float(timestamp_str)
                # Handle milliseconds vs seconds
                if ts > 1e10:  # Likely milliseconds
                    ts /= 1000
                return datetime.datetime.fromtimestamp(ts)
        except ValueError:
            pass
            
        return None

    def matches(self, timestamp: Union[str, datetime.datetime, None]) -> bool:
        """
        Check if a timestamp is within the filter range.
        
        Args:
            timestamp: Timestamp to check (string, datetime, or None)
            
        Returns:
            True if timestamp is within range or no filter is active
        """
        # If no time filter set, all timestamps match
        if not self.start_date and not self.end_date:
            return True
            
        # If timestamp is None, can't apply filter
        if timestamp is None:
            return False
            
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            dt_timestamp = self.parse_timestamp(timestamp)
            if dt_timestamp is None:
                return False
        else:
            dt_timestamp = timestamp
            
        # Apply filter
        if self.start_date and dt_timestamp < self.start_date:
            return False
        if self.end_date and dt_timestamp > self.end_date:
            return False
            
        return True


class ErrorHandler:
    """
    Handles errors during log processing according to specified strategy.
    """
    
    # Error handling strategies
    SKIP = "skip"  # Skip the file and continue
    WARN = "warn"  # Log a warning and continue
    FAIL = "fail"  # Raise an exception
    
    def __init__(self, strategy: str = SKIP):
        """
        Initialize the error handler.
        
        Args:
            strategy: Error handling strategy (skip, warn, fail)
        """
        if strategy not in [self.SKIP, self.WARN, self.FAIL]:
            logger.warning(f"Unknown error handling strategy: {strategy}, defaulting to 'skip'")
            self.strategy = self.SKIP
        else:
            self.strategy = strategy.lower()
    
    def handle(self, error_msg: str, exception: Optional[Exception] = None) -> None:
        """
        Handle an error according to the specified strategy.
        
        Args:
            error_msg: Error message
            exception: Optional exception object
            
        Raises:
            ValueError: If strategy is 'fail'
        """
        if self.strategy == self.FAIL:
            if exception:
                raise exception
            else:
                raise ValueError(error_msg)
        elif self.strategy == self.WARN:
            logger.warning(error_msg)
        else:  # SKIP
            logger.error(error_msg)


class FormatDetector:
    """
    Detects the format of log files.
    """
    
    # Supported formats
    SQL = "sql"
    JSON = "json"
    CSV = "csv"
    
    @classmethod
    def detect_format(cls, file_path: Path) -> str:
        """
        Detect the format of a log file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Detected format (sql, json, csv)
        """
        # Check file extension
        suffix = file_path.suffix.lower()
        if suffix == ".sql":
            return cls.SQL
        elif suffix in [".json", ".jsonl"]:
            return cls.JSON
        elif suffix == ".csv":
            return cls.CSV
        
        # Try to read the first few lines
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = "".join(f.readline() for _ in range(5))
            
            # Check for SQL patterns
            if re.search(r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s", sample, re.IGNORECASE):
                return cls.SQL
            
            # Check for JSON patterns
            if (sample.strip().startswith("{") and "}" in sample) or \
               (sample.strip().startswith("[") and "]" in sample):
                return cls.JSON
            
            # Check for CSV patterns
            if "," in sample and len(sample.split("\n")[0].split(",")) > 1:
                return cls.CSV
            
        except Exception as e:
            logger.warning(f"Error detecting format for {file_path}: {e}")
        
        # Default to SQL if detection fails
        logger.warning(f"Could not detect format for {file_path}, assuming SQL")
        return cls.SQL


class SQLProcessor:
    """
    Processes SQL log files.
    """
    
    def __init__(self, time_filter: TimeFilter, error_handler: ErrorHandler):
        """
        Initialize the SQL processor.
        
        Args:
            time_filter: Time filter for filtering log entries
            error_handler: Error handler for processing errors
        """
        self.time_filter = time_filter
        self.error_handler = error_handler
    
    def process(self, file_path: Path) -> List[LogEntry]:
        """
        Process an SQL log file.
        
        Args:
            file_path: Path to the SQL log file
            
        Returns:
            List of LogEntry objects
        """
        results = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Parse SQL statements
            statements = sqlparse.split(content)
            
            for i, stmt in enumerate(statements):
                if not stmt.strip():
                    continue
                
                try:
                    parsed = sqlparse.parse(stmt)[0]
                    stmt_type = parsed.get_type() if parsed.get_type() else "UNKNOWN"
                    
                    # Extract timestamp if present in comments
                    timestamp = None
                    timestamp_str = None
                    timestamp_match = re.search(r'-- (\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})', stmt)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        timestamp = self.time_filter.parse_timestamp(timestamp_str)
                    
                    # Apply time filter if timestamp is found
                    if timestamp and not self.time_filter.matches(timestamp):
                        continue
                    
                    # Extract table names
                    tables = []
                    for token in parsed.tokens:
                        if token.ttype is None and hasattr(token, 'get_real_name'):
                            if token.get_real_name() and token.get_real_name() not in tables:
                                tables.append(token.get_real_name())
                    
                    # Extract schema details
                    schema_name = None
                    for token in parsed.tokens:
                        if token.ttype is None and hasattr(token, 'get_parent_name'):
                            schema_name = token.get_parent_name()
                            if schema_name:
                                break
                    
                    # Preserve SQL query structure and formatting
                    formatted_query = sqlparse.format(
                        stmt,
                        reindent=True,
                        keyword_case='upper',
                        identifier_case='lower',
                        strip_comments=False
                    )
                    
                    # Create a log entry
                    entry = LogEntry(
                        source_file=file_path.name,
                        source_format=FormatDetector.SQL,
                        log_type=stmt_type,
                        timestamp=timestamp.isoformat() if timestamp else None,
                        raw_content=stmt,
                        processed_content={
                            "formatted_query": formatted_query,
                            "tables": tables,
                            "schema": schema_name
                        },
                        metadata={
                            "query_length": len(stmt),
                            "tables_referenced": len(tables)
                        },
                        processing_info={
                            "index": i,
                            "timestamp_source": "comment" if timestamp else None
                        }
                    )
                    
                    results.append(entry)
                except Exception as e:
                    self.error_handler.handle(
                        f"Error processing SQL statement {i} in {file_path}: {e}",
                        exception=e
                    )
            
            logger.info(f"Processed SQL log: {file_path}, found {len(results)} statements")
            return results
            
        except Exception as e:
            self.error_handler.handle(
                f"Error processing SQL log {file_path}: {e}",
                exception=e
            )
            return []


class JSONProcessor:
    """
    Processes JSON log files.
    """
    
    def __init__(self, time_filter: TimeFilter, error_handler: ErrorHandler):
        """
        Initialize the JSON processor.
        
        Args:
            time_filter: Time filter for filtering log entries
            error_handler: Error handler for processing errors
        """
        self.time_filter = time_filter
        self.error_handler = error_handler
        
        # Common timestamp field names
        self.timestamp_fields = [
            "timestamp", "time", "date", "created_at", "datetime", 
            "createdAt", "created", "eventTime", "logTime", "@timestamp"
        ]
    
    def process(self, file_path: Path) -> List[LogEntry]:
        """
        Process a JSON log file.
        
        Args:
            file_path: Path to the JSON log file
            
        Returns:
            List of LogEntry objects
        """
        results = []
        
        try:
            # Check if it's a single JSON object or multiple JSON lines
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == "[":
                    # Single JSON array
                    try:
                        data = json.load(f)
                        entries = data if isinstance(data, list) else [data]
                    except json.JSONDecodeError as e:
                        self.error_handler.handle(
                            f"Error parsing JSON file {file_path}: {e}",
                            exception=e
                        )
                        return []
                else:
                    # JSON lines
                    entries = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                self.error_handler.handle(
                                    f"Error parsing JSON line {line_num} in {file_path}: {line[:100]}...",
                                    exception=e
                                )
            
            # Process each entry
            for i, entry in enumerate(entries):
                # Try to extract timestamp
                timestamp = None
                timestamp_str = None
                timestamp_field = None
                
                for ts_field in self.timestamp_fields:
                    if ts_field in entry and entry[ts_field] is not None:
                        timestamp_str = entry[ts_field]
                        timestamp = self.time_filter.parse_timestamp(str(timestamp_str))
                        if timestamp:
                            timestamp_field = ts_field
                            break
                
                # Apply time filter if timestamp is found
                if timestamp and not self.time_filter.matches(timestamp):
                    continue
                
                # Extract log type if present
                log_type = "JSON"
                for type_field in ["type", "event", "level", "severity", "category"]:
                    if type_field in entry and entry[type_field]:
                        log_type = str(entry[type_field])
                        break
                
                # Create processed content (a copy of the original)
                processed_content = dict(entry)
                
                # Create a log entry
                log_entry = LogEntry(
                    source_file=file_path.name,
                    source_format=FormatDetector.JSON,
                    log_type=log_type,
                    timestamp=timestamp.isoformat() if timestamp else None,
                    raw_content=json.dumps(entry, ensure_ascii=False),
                    processed_content=processed_content,
                    metadata={
                        "field_count": len(entry),
                        "has_nested_objects": any(isinstance(v, dict) for v in entry.values())
                    },
                    processing_info={
                        "index": i,
                        "timestamp_field": timestamp_field,
                    }
                )
                
                results.append(log_entry)
            
            logger.info(f"Processed JSON log: {file_path}, found {len(results)} entries")
            return results
            
        except Exception as e:
            self.error_handler.handle(
                f"Error processing JSON log {file_path}: {e}",
                exception=e
            )
            return []


class CSVProcessor:
    """
    Processes CSV log files.
    """
    
    def __init__(self, time_filter: TimeFilter, error_handler: ErrorHandler):
        """
        Initialize the CSV processor.
        
        Args:
            time_filter: Time filter for filtering log entries
            error_handler: Error handler for processing errors
        """
        self.time_filter = time_filter
        self.error_handler = error_handler
        
        # Common timestamp field names
        self.timestamp_fields = [
            "timestamp", "time", "date", "created_at", "datetime", 
            "createdAt", "created", "eventTime", "logTime"
        ]
    
    def process(self, file_path: Path) -> List[LogEntry]:
        """
        Process a CSV log file.
        
        Args:
            file_path: Path to the CSV log file
            
        Returns:
            List of LogEntry objects
        """
        results = []
        
        try:
            # Try different delimiters and encodings
            try:
                # Use pandas with automatic dialect detection
                df = pd.read_csv(file_path, engine="python")
            except Exception as e:
                # Try with explicit delimiter detection
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    sample = f.readline() + f.readline()
                
                dialect = csv.Sniffer().sniff(sample)
                df = pd.read_csv(file_path, sep=dialect.delimiter)
            
            # Convert to list of dictionaries
            raw_entries = df.to_dict(orient="records")
            
            # Process each entry
            for i, entry in enumerate(raw_entries):
                # Convert any NaN or non-serializable values
                processed_entry = {}
                for k, v in entry.items():
                    if pd.isna(v):
                        processed_entry[k] = None
                    elif isinstance(v, (int, float, bool, str)) or v is None:
                        processed_entry[k] = v
                    else:
                        processed_entry[k] = str(v)
                
                # Try to extract timestamp
                timestamp = None
                timestamp_str = None
                timestamp_field = None
                
                for ts_field in self.timestamp_fields:
                    if ts_field in processed_entry and processed_entry[ts_field] is not None:
                        timestamp_str = processed_entry[ts_field]
                        timestamp = self.time_filter.parse_timestamp(str(timestamp_str))
                        if timestamp:
                            timestamp_field = ts_field
                            break
                
                # Apply time filter if timestamp is found
                if timestamp and not self.time_filter.matches(timestamp):
                    continue
                
                # Extract log type if present
                log_type = "CSV"
                for type_field in ["type", "event", "level", "severity", "category"]:
                    if type_field in processed_entry and processed_entry[type_field]:
                        log_type = str(processed_entry[type_field])
                        break
                
                # Create the raw content as CSV row
                raw_content = ",".join(str(entry.get(col, "")) for col in df.columns)
                
                # Create a log entry
                log_entry = LogEntry(
                    source_file=file_path.name,
                    source_format=FormatDetector.CSV,
                    log_type=log_type,
                    timestamp=timestamp.isoformat() if timestamp else None,
                    raw_content=raw_content,
                    processed_content=processed_entry,
                    metadata={
                        "column_count": len(df.columns),
                        "columns": list(df.columns)
                    },
                    processing_info={
                        "index": i,
                        "timestamp_field": timestamp_field,
                    }
                )
                
                results.append(log_entry)
            
            logger.info(f"Processed CSV log: {file_path}, found {len(results)} entries")
            return results
            
        except Exception as e:
            self.error_handler.handle(
                f"Error processing CSV log {file_path}: {e}",
                exception=e
            )
            return []


class DBLogProcessor:
    """
    Main processor for database logs, handling various formats.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        format: str = "auto",
        time_filter: Dict[str, str] = None,
        error_handling: str = "skip",
        output_format: str = "json",
        preserve_structure: bool = True,
    ):
        """
        Initialize the database log processor.
        
        Args:
            input_dir: Directory containing log files
            output_dir: Directory to save processed logs
            format: Log format (auto, sql, json, csv)
            time_filter: Dictionary with start_date and end_date
            error_handling: How to handle errors (skip, warn, fail)
            output_format: Output format (json, jsonl)
            preserve_structure: Whether to preserve log structure
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.output_format = output_format.lower()
        self.preserve_structure = preserve_structure
        
        # Set up time filter
        time_filter = time_filter or {}
        self.time_filter = TimeFilter(
            start_date=time_filter.get("start_date"),
            end_date=time_filter.get("end_date")
        )
        
        # Set up error handler
        self.error_handler = ErrorHandler(error_handling)
        
        # Set up format-specific processors
        self.sql_processor = SQLProcessor(self.time_filter, self.error_handler)
        self.json_processor = JSONProcessor(self.time_filter, self.error_handler)
        self.csv_processor = CSVProcessor(self.time_filter, self.error_handler)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_file(self, file_path: Path) -> List[LogEntry]:
        """
        Process a single log file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List of LogEntry objects
        """
        # Detect or use specified format
        format_to_use = self.format if self.format != "auto" else FormatDetector.detect_format(file_path)
        
        # Process based on format
        if format_to_use == FormatDetector.SQL:
            return self.sql_processor.process(file_path)
        elif format_to_use == FormatDetector.JSON:
            return self.json_processor.process(file_path)
        elif format_to_use == FormatDetector.CSV:
            return self.csv_processor.process(file_path)
        else:
            logger.error(f"Unsupported format: {format_to_use}")
            return []
    
    def save_entries(self, entries: List[LogEntry], output_path: Path) -> None:
        """
        Save log entries to a file.
        
        Args:
            entries: List of LogEntry objects
            output_path: Path to save the file
        """
        if self.output_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.to_json(indent=None) + "\n")
        else:  # json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([entry.to_dict() for entry in entries], f, ensure_ascii=False, indent=2, default=str)
    
    def process_all(self) -> Dict[str, Any]:
        """
        Process all log files in the input directory.
        
        Returns:
            Dictionary with stats on processing results
        """
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_entries": 0,
            "entries_by_format": {
                FormatDetector.SQL: 0,
                FormatDetector.JSON: 0,
                FormatDetector.CSV: 0
            },
            "errors": 0,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None
        }
        
        # Find all files in the input directory
        files = list(self.input_dir.glob("*.*"))
        stats["total_files"] = len(files)
        
        # Process each file
        for file_path in tqdm(files, desc="Processing log files"):
            try:
                # Skip hidden files
                if file_path.name.startswith("."):
                    continue
                
                # Process the file
                entries = self.process_file(file_path)
                
                if entries:
                    # Save the processed entries
                    output_path = self.output_dir / f"{file_path.stem}_processed.{self.output_format}"
                    self.save_entries(entries, output_path)
                    
                    # Update stats
                    stats["processed_files"] += 1
                    stats["total_entries"] += len(entries)
                    
                    # Update format-specific stats
                    if entries and entries[0].source_format:
                        format_key = entries[0].source_format
                        stats["entries_by_format"][format_key] = stats["entries_by_format"].get(format_key, 0) + len(entries)
                else:
                    stats["errors"] += 1
                    
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing {file_path}: {e}")
        
        # Update end time
        stats["end_time"] = datetime.datetime.now().isoformat()
        stats["duration_seconds"] = (datetime.datetime.fromisoformat(stats["end_time"]) - 
                                     datetime.datetime.fromisoformat(stats["start_time"])).total_seconds()
        
        logger.info(f"DB Log processing completed. Stats: {stats}")
        
        # Save stats
        with open(self.output_dir / "processing_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        
        return stats


def main():
    """Main function for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Process database logs for Llama fine-tuning")
    parser.add_argument("--input_dir", required=True, help="Directory containing log files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed logs")
    parser.add_argument("--format", default="auto", choices=["auto", "sql", "json", "csv"], 
                        help="Log format (auto, sql, json, csv)")
    parser.add_argument("--start_date", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--error_handling", default="skip", choices=["skip", "warn", "fail"],
                        help="How to handle errors (skip, warn, fail)")
    parser.add_argument("--output_format", default="json", choices=["json", "jsonl"],
                        help="Output format (json, jsonl)")
    parser.add_argument("--preserve_structure", action="store_true",
                        help="Preserve log structure in output")

    args = parser.parse_args()
    
    # Set up time filter
    time_filter = {
        "start_date": args.start_date,
        "end_date": args.end_date
    }
    
    processor = DBLogProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format=args.format,
        time_filter=time_filter,
        error_handling=args.error_handling,
        output_format=args.output_format,
        preserve_structure=args.preserve_structure
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()
