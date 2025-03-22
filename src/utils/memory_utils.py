"""
Memory utilities for QLoRA fine-tuning in memory-constrained environments.
Provides functions for monitoring GPU memory usage and cleaning memory.
"""

import os
import logging
import gc
from typing import Dict, Union, Optional

import torch

logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Dict[str, Union[int, float, str]]:
    """
    Get detailed GPU memory information.
    
    Returns:
        Dictionary with GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": 0,
            "memory_info": []
        }
    
    device_count = torch.cuda.device_count()
    memory_info = []
    
    for device_idx in range(device_count):
        with torch.cuda.device(device_idx):
            device_name = torch.cuda.get_device_name(device_idx)
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory
            reserved_memory = torch.cuda.memory_reserved(device_idx)
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            free_memory = total_memory - allocated_memory
            
            # Convert to MB for readability
            total_memory_mb = total_memory / (1024 ** 2)
            reserved_memory_mb = reserved_memory / (1024 ** 2)
            allocated_memory_mb = allocated_memory / (1024 ** 2)
            free_memory_mb = free_memory / (1024 ** 2)
            
            device_info = {
                "device_idx": device_idx,
                "device_name": device_name,
                "total_memory_mb": total_memory_mb,
                "reserved_memory_mb": reserved_memory_mb,
                "allocated_memory_mb": allocated_memory_mb,
                "free_memory_mb": free_memory_mb,
                "utilization_percent": (allocated_memory / total_memory) * 100
            }
            memory_info.append(device_info)
    
    return {
        "available": True,
        "device_count": device_count,
        "memory_info": memory_info
    }

def print_gpu_memory_summary(device_id: int = 0) -> None:
    """
    Print a summary of GPU memory usage.
    
    Args:
        device_id: CUDA device ID
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot print GPU memory summary")
        return
    
    with torch.cuda.device(device_id):
        device_name = torch.cuda.get_device_name(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        free_memory = (total_memory - allocated_memory)
        
        logger.info(f"GPU {device_id} ({device_name}):")
        logger.info(f"  Total memory: {total_memory:.2f} MB")
        logger.info(f"  Reserved by PyTorch: {reserved_memory:.2f} MB")
        logger.info(f"  Allocated: {allocated_memory:.2f} MB")
        logger.info(f"  Free: {free_memory:.2f} MB")
        logger.info(f"  Utilization: {(allocated_memory / total_memory) * 100:.2f}%")

def clean_memory() -> float:
    """
    Clean GPU memory by garbage collection and emptying CUDA cache.
    
    Returns:
        Amount of memory freed in MB
    """
    if not torch.cuda.is_available():
        return 0
    
    # Record memory before cleaning
    before = torch.cuda.memory_allocated()
    
    # Run garbage collection
    gc.collect()
    
    # Empty CUDA cache
    torch.cuda.empty_cache()
    
    # Record memory after cleaning
    after = torch.cuda.memory_allocated()
    
    # Calculate memory freed (in MB)
    memory_freed = (before - after) / (1024 ** 2)
    
    logger.info(f"Cleaned GPU memory: {memory_freed:.2f} MB freed")
    return memory_freed

def get_device_with_most_memory() -> Optional[torch.device]:
    """
    Get the GPU device with the most available memory.
    
    Returns:
        Torch device with the most memory, or None if no GPU is available
    """
    if not torch.cuda.is_available():
        return None
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None
    
    # Find device with most free memory
    max_free_memory = 0
    best_device = 0
    
    for device_idx in range(device_count):
        with torch.cuda.device(device_idx):
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            free_memory = total_memory - allocated_memory
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device = device_idx
    
    return torch.device(f"cuda:{best_device}")
