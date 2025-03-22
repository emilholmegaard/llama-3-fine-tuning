"""
Memory tracking utilities for QLoRA fine-tuning in memory-constrained environments.
Provides a MemoryTracker class for monitoring GPU memory during training.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

class MemoryTracker:
    """
    Class for tracking GPU memory during training.
    """
    
    def __init__(self, log_dir: str = "memory_logs", log_interval: int = 10):
        """
        Initialize memory tracker.
        
        Args:
            log_dir: Directory to save memory logs
            log_interval: How often to log memory statistics (in steps)
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.memory_stats = []
        self.step_counter = 0
        self.peak_memory = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"Initialized memory tracker, logging to {log_dir}")
    
    def start_tracking(self) -> None:
        """Start or reset memory tracking."""
        self.memory_stats = []
        self.step_counter = 0
        self.peak_memory = 0
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def update(self, step: Optional[int] = None, force_log: bool = False) -> Dict[str, Any]:
        """
        Update memory tracking with current state.
        
        Args:
            step: Current training step (if None, uses internal counter)
            force_log: Whether to force logging regardless of interval
            
        Returns:
            Current memory statistics
        """
        if not torch.cuda.is_available():
            return {}
        
        # Use provided step or increment internal counter
        current_step = step if step is not None else self.step_counter
        self.step_counter = current_step + 1
        
        # Get memory statistics
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        # Update peak memory
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        self.peak_memory = max(self.peak_memory, peak_allocated)
        
        memory_stat = {
            "step": current_step,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_allocated_mb": peak_allocated,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add to stats list
        self.memory_stats.append(memory_stat)
        
        # Log if at interval or forced
        if force_log or current_step % self.log_interval == 0:
            logger.info(f"Memory at step {current_step}: {allocated:.2f}MB allocated, "
                        f"{peak_allocated:.2f}MB peak, {reserved:.2f}MB reserved")
        
        return memory_stat
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory usage during tracked period.
        
        Returns:
            Dictionary with memory usage summary
        """
        if not self.memory_stats:
            return {
                "tracked_steps": 0,
                "peak_memory_mb": 0,
                "average_memory_mb": 0,
            }
        
        # Calculate statistics
        allocated_values = [stat["allocated_mb"] for stat in self.memory_stats]
        peak_values = [stat["peak_allocated_mb"] for stat in self.memory_stats]
        
        summary = {
            "tracked_steps": len(self.memory_stats),
            "peak_memory_mb": self.peak_memory,
            "average_memory_mb": sum(allocated_values) / len(allocated_values),
            "min_memory_mb": min(allocated_values),
            "max_memory_mb": max(allocated_values),
            "final_memory_mb": allocated_values[-1] if allocated_values else 0,
        }
        
        return summary
    
    def save_log(self, filename: Optional[str] = None) -> str:
        """
        Save memory log to file.
        
        Args:
            filename: Filename to save to (default: memory_log_{timestamp}.json)
            
        Returns:
            Path to saved log file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_log_{timestamp}.json"
        
        log_path = os.path.join(self.log_dir, filename)
        
        # Add summary to log
        log_data = {
            "stats": self.memory_stats,
            "summary": self.get_summary()
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Saved memory log to {log_path}")
        return log_path
