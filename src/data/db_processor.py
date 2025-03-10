"""
Database log processor for Llama 3.3 fine-tuning.

This module extracts and processes content from database log files in various formats
(SQL, JSON, CSV) for use in fine-tuning.
"""

import os
import logging
import json
import csv
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import sqlparse
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DBLogProcessor:
    """Process database logs for fine-tuning."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        format: str = "auto",
        time_filter: Dict[str, str] = None,
        error_handling: str = "skip",
    ):
        """
        Initialize the database log processor.

        Args:
            input_dir: Directory containing log files
            output_dir: Directory to save processed logs
            format: Log format (auto, sql, json, csv)
            time_filter: Dictionary with start_date and end_date
            error_handling: How to handle errors (skip, warn, fail)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.time_filter = time_filter or {"start_date": None, "end_date": None}
        self.error_handling = error_handling.lower()

        # Convert date strings to datetime objects
        if self.time_filter["start_date"]:
            self.time_filter["start_date"] = datetime.datetime.strptime(
                self.time_filter["start_date"], "%Y-%m-%d"
            )
        if self.time_filter["end_date"]:
            self.time_filter["end_date"] = datetime.datetime.strptime(
                self.time_filter["end_date"], "%Y-%m-%d"
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_format(self, file_path: Path) -> str:
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
            return "sql"
        elif suffix == ".json" or suffix == ".jsonl":
            return "json"
        elif suffix == ".csv":
            return "csv"

        # Try to read the first few lines
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sample = "".join(f.readline() for _ in range(5))

            # Check for SQL patterns
            if re.search(r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s", sample, re.IGNORECASE):
                return "sql"
            
            # Check for JSON patterns
            if (sample.strip().startswith("{") and sample.strip().endswith("}")) or \
               (sample.strip().startswith("[") and sample.strip().endswith("]")):
                return "json"
            
            # Check for CSV patterns
            if "," in sample and len(sample.split("\n")[0].split(",")) > 1:
                return "csv"
            
        except Exception as e:
            logger.warning(f"Error detecting format for {file_path}: {e}")
        
        # Default to SQL if detection fails
        logger.warning(f"Could not detect format for {file_path}, assuming SQL")
        return "sql"

    def process_sql_log(self, file_path: Path) -> List[Dict]:
        """
        Process an SQL log file.

        Args:
            file_path: Path to the SQL log file

        Returns:
            List of dictionaries representing processed log entries
        """
        results = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse SQL statements
            statements = sqlparse.split(content)
            
            for i, stmt in enumerate(statements):
                if not stmt.strip():
                    continue
                
                parsed = sqlparse.parse(stmt)[0]
                stmt_type = parsed.get_type() if parsed.get_type() else "UNKNOWN"
                
                # Extract timestamp if present in comments
                timestamp = None
                timestamp_match = re.search(r'-- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', stmt)
                if timestamp_match:
                    try:
                        timestamp = datetime.datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                # Apply time filter if timestamp is found
                if timestamp and self.time_filter["start_date"] and timestamp < self.time_filter["start_date"]:
                    continue
                if timestamp and self.time_filter["end_date"] and timestamp > self.time_filter["end_date"]:
                    continue
                
                # Extract table names
                tables = []
                for token in parsed.tokens:
                    if token.ttype is None and hasattr(token, 'get_name'):
                        if token.get_name() and token.get_name() not in tables:
                            tables.append(token.get_name())
                
                # Create a log entry
                entry = {
                    "type": stmt_type,
                    "query": stmt,
                    "formatted_query": sqlparse.format(stmt, reindent=True, keyword_case='upper'),
                    "tables": tables,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "file": file_path.name,
                    "index": i
                }
                
                results.append(entry)
            
            logger.info(f"Processed SQL log: {file_path}, found {len(results)} statements")
            return results
            
        except Exception as e:
            error_msg = f"Error processing SQL log {file_path}: {e}"
            if self.error_handling == "fail":
                raise ValueError(error_msg)
            elif self.error_handling == "warn":
                logger.warning(error_msg)
            else:
                logger.error(error_msg)
            return []

    def process_json_log(self, file_path: Path) -> List[Dict]:
        """
        Process a JSON log file.

        Args:
            file_path: Path to the JSON log file

        Returns:
            List of dictionaries representing processed log entries
        """
        results = []
        
        try:
            # Check if it's a single JSON object or multiple JSON lines
            with open(file_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == "[":
                    # Single JSON array
                    data = json.load(f)
                    entries = data if isinstance(data, list) else [data]
                else:
                    # JSON lines
                    entries = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                if self.error_handling == "fail":
                                    raise
                                else:
                                    logger.warning(f"Error parsing JSON line: {line[:100]}...")
            
            # Process each entry
            for i, entry in enumerate(entries):
                # Try to extract timestamp
                timestamp = None
                for ts_field in ["timestamp", "time", "date", "created_at", "datetime"]:
                    if ts_field in entry:
                        try:
                            if isinstance(entry[ts_field], str):
                                # Try different date formats
                                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                    try:
                                        timestamp = datetime.datetime.strptime(entry[ts_field], fmt)
                                        break
                                    except ValueError:
                                        continue
                            elif isinstance(entry[ts_field], (int, float)):
                                # Unix timestamp
                                timestamp = datetime.datetime.fromtimestamp(entry[ts_field])
                        except:
                            pass
                        if timestamp:
                            break
                
                # Apply time filter if timestamp is found
                if timestamp and self.time_filter["start_date"] and timestamp < self.time_filter["start_date"]:
                    continue
                if timestamp and self.time_filter["end_date"] and timestamp > self.time_filter["end_date"]:
                    continue
                
                # Add metadata
                result = entry.copy()
                result["_file"] = file_path.name
                result["_index"] = i
                result["_processed_timestamp"] = datetime.datetime.now().isoformat()
                
                results.append(result)
            
            logger.info(f"Processed JSON log: {file_path}, found {len(results)} entries")
            return results
            
        except Exception as e:
            error_msg = f"Error processing JSON log {file_path}: {e}"
            if self.error_handling == "fail":
                raise ValueError(error_msg)
            elif self.error_handling == "warn":
                logger.warning(error_msg)
            else:
                logger.error(error_msg)
            return []

    def process_csv_log(self, file_path: Path) -> List[Dict]:
        """
        Process a CSV log file.

        Args:
            file_path: Path to the CSV log file

        Returns:
            List of dictionaries representing processed log entries
        """
        results = []
        
        try:
            # Use pandas to handle various CSV formats
            df = pd.read_csv(file_path, engine="python")
            
            # Convert to list of dictionaries
            entries = df.to_dict(orient="records")
            
            # Process each entry
            for i, entry in enumerate(entries):
                # Try to extract timestamp
                timestamp = None
                for ts_field in ["timestamp", "time", "date", "created_at", "datetime"]:
                    if ts_field in entry and entry[ts_field] is not None:
                        try:
                            if isinstance(entry[ts_field], str):
                                # Try different date formats
                                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                    try:
                                        timestamp = datetime.datetime.strptime(entry[ts_field], fmt)
                                        break
                                    except ValueError:
                                        continue
                            elif isinstance(entry[ts_field], (int, float)):
                                # Unix timestamp
                                timestamp = datetime.datetime.fromtimestamp(entry[ts_field])
                        except:
                            pass
                        if timestamp:
                            break
                
                # Apply time filter if timestamp is found
                if timestamp and self.time_filter["start_date"] and timestamp < self.time_filter["start_date"]:
                    continue
                if timestamp and self.time_filter["end_date"] and timestamp > self.time_filter["end_date"]:
                    continue
                
                # Convert any non-serializable objects to strings
                processed_entry = {}
                for k, v in entry.items():
                    if pd.isna(v):
                        processed_entry[k] = None
                    else:
                        processed_entry[k] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                
                # Add metadata
                processed_entry["_file"] = file_path.name
                processed_entry["_index"] = i
                processed_entry["_processed_timestamp"] = datetime.datetime.now().isoformat()
                
                results.append(processed_entry)
            
            logger.info(f"Processed CSV log: {file_path}, found {len(results)} entries")
            return results
            
        except Exception as e:
            error_msg = f"Error processing CSV log {file_path}: {e}"
            if self.error_handling == "fail":
                raise ValueError(error_msg)
            elif self.error_handling == "warn":
                logger.warning(error_msg)
            else:
                logger.error(error_msg)
            return []

    def process_file(self, file_path: Path) -> List[Dict]:
        """
        Process a single log file.

        Args:
            file_path: Path to the log file

        Returns:
            List of dictionaries representing processed log entries
        """
        format_to_use = self.format if self.format != "auto" else self.detect_format(file_path)
        
        if format_to_use == "sql":
            return self.process_sql_log(file_path)
        elif format_to_use == "json":
            return self.process_json_log(file_path)
        elif format_to_use == "csv":
            return self.process_csv_log(file_path)
        else:
            logger.error(f"Unsupported format: {format_to_use}")
            return []

    def process_all(self) -> Dict[str, int]:
        """
        Process all log files in the input directory.

        Returns:
            Dictionary with stats on processing results
        """
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_entries": 0,
            "errors": 0
        }
        
        # Find all files in the input directory
        files = list(self.input_dir.glob("*.*"))
        stats["total_files"] = len(files)
        
        for file_path in tqdm(files, desc="Processing log files"):
            try:
                entries = self.process_file(file_path)
                
                if entries:
                    # Save the processed entries
                    output_path = self.output_dir / f"{file_path.stem}_processed.json"
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(entries, f, ensure_ascii=False, indent=2)
                    
                    stats["processed_files"] += 1
                    stats["total_entries"] += len(entries)
                else:
                    stats["errors"] += 1
                    
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"DB Log processing completed. Stats: {stats}")
        
        # Save stats
        with open(self.output_dir / "processing_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
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
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()
