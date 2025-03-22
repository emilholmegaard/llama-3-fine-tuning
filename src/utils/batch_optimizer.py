"""
Batch optimization utilities for QLoRA fine-tuning in memory-constrained environments.
Provides functionality for finding optimal batch sizes and implementing microbatching.
"""

import os
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.memory_utils import clean_memory

logger = logging.getLogger(__name__)

def find_optimal_batch_size(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    start_batch_size: int = 16,
    max_sequence_length: int = 2048,
    min_batch_size: int = 1,
    max_memory_usage_pct: float = 0.9,
    step_factor: float = 0.5
) -> int:
    """
    Find the optimal maximum batch size that fits in GPU memory.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer used with the model
        start_batch_size: Starting batch size to try
        max_sequence_length: Maximum sequence length for samples
        min_batch_size: Minimum acceptable batch size
        max_memory_usage_pct: Maximum GPU memory usage percentage
        step_factor: Factor to reduce batch size by when OOM occurs
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, defaulting to minimum batch size")
        return min_batch_size
    
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    max_memory = total_memory * max_memory_usage_pct
    
    # Clean up memory before testing
    clean_memory()
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Create a test input tensor
    batch_size = start_batch_size
    test_tokens = None
    
    logger.info(f"Finding optimal batch size (starting at {batch_size}, min: {min_batch_size})")
    
    while batch_size >= min_batch_size:
        try:
            # Generate dummy inputs
            dummy_input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, max_sequence_length), 
                device=device, dtype=torch.long
            )
            dummy_attention_mask = torch.ones(
                (batch_size, max_sequence_length), 
                device=device, dtype=torch.long
            )
            
            # Forward pass
            with torch.no_grad():
                _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            
            # Check memory usage
            allocated_memory = torch.cuda.memory_allocated()
            if allocated_memory < max_memory:
                # This batch size works
                logger.info(f"Batch size {batch_size} works, using {allocated_memory / (1024**2):.2f}MB out of {max_memory / (1024**2):.2f}MB")
                
                # Clean up testing tensors
                del dummy_input_ids, dummy_attention_mask
                clean_memory()
                
                return batch_size
        
        except torch.cuda.OutOfMemoryError:
            # Ran out of memory
            logger.info(f"Batch size {batch_size} caused OOM, reducing...")
            
            # Clean up after OOM
            clean_memory()
        
        # Reduce batch size
        new_batch_size = max(min_batch_size, int(batch_size * step_factor))
        if new_batch_size == batch_size:
            # Cannot reduce further, break
            batch_size = new_batch_size - 1
        else:
            batch_size = new_batch_size
    
    # If we've reached here, even the minimum batch size is too large
    logger.warning(f"Even minimum batch size {min_batch_size} is too large, consider further optimizations")
    return min_batch_size


class MicroBatchingDataloader:
    """
    Implements microbatching for memory-efficient training.
    Processes a large batch in smaller chunks to reduce memory usage.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        micro_batch_size: int,
        accumulation_steps: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize microbatching dataloader.
        
        Args:
            dataloader: Original dataloader
            micro_batch_size: Size of microbatches to process
            accumulation_steps: Number of gradient accumulation steps
            device: Device to place tensors on (if None, uses CUDA if available)
        """
        self.dataloader = dataloader
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = accumulation_steps
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._iterator = None
    
    def __iter__(self):
        """Get iterator for dataloader."""
        self._iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        """Get next batch of microbatches."""
        try:
            batch = next(self._iterator)
            
            # Split the batch into microbatches
            microbatches = []
            batch_size = self._get_batch_size(batch)
            
            if batch_size <= self.micro_batch_size:
                # Batch is smaller than micro_batch_size, just move to device
                batch = self._to_device(batch)
                microbatches.append(batch)
            else:
                # Split the batch into microbatches
                num_microbatches = math.ceil(batch_size / self.micro_batch_size)
                
                for i in range(num_microbatches):
                    start_idx = i * self.micro_batch_size
                    end_idx = min((i + 1) * self.micro_batch_size, batch_size)
                    
                    microbatch = self._extract_slice(batch, start_idx, end_idx)
                    microbatch = self._to_device(microbatch)
                    microbatches.append(microbatch)
            
            return microbatches
        
        except StopIteration:
            raise StopIteration
    
    def _get_batch_size(self, batch) -> int:
        """Get batch size from batch."""
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
        elif isinstance(batch, list):
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                return batch[0].size(0)
        elif isinstance(batch, torch.Tensor):
            return batch.size(0)
        
        # Default
        return 1
    
    def _extract_slice(self, batch, start_idx: int, end_idx: int):
        """Extract slice from batch."""
        if isinstance(batch, dict):
            return {k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, list):
            return [v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch[start_idx:end_idx]
        else:
            return batch
    
    def _to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, list):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch


class DynamicGradientAccumulation:
    """
    Dynamic gradient accumulation for memory-efficient training.
    Adjusts gradient accumulation steps based on available memory.
    """
    
    def __init__(
        self,
        base_batch_size: int,
        base_accumulation_steps: int,
        min_batch_size: int = 1,
        max_accumulation_steps: int = 32
    ):
        """
        Initialize dynamic gradient accumulation.
        
        Args:
            base_batch_size: Base batch size for calculations
            base_accumulation_steps: Base accumulation steps
            min_batch_size: Minimum allowed batch size
            max_accumulation_steps: Maximum allowed accumulation steps
        """
        self.base_batch_size = base_batch_size
        self.base_accumulation_steps = base_accumulation_steps
        self.min_batch_size = min_batch_size
        self.max_accumulation_steps = max_accumulation_steps
    
    def adjust_for_batch_size(self, actual_batch_size: int) -> int:
        """
        Calculate adjusted gradient accumulation steps for a given batch size.
        Ensures the effective batch size remains constant.
        
        Args:
            actual_batch_size: Actual batch size being used
            
        Returns:
            Adjusted gradient accumulation steps
        """
        if actual_batch_size >= self.base_batch_size:
            # If we can use a larger batch size, reduce accumulation steps
            ratio = actual_batch_size / self.base_batch_size
            new_steps = max(1, int(self.base_accumulation_steps / ratio))
        else:
            # If we need to use a smaller batch size, increase accumulation steps
            ratio = self.base_batch_size / actual_batch_size
            new_steps = min(self.max_accumulation_steps, int(self.base_accumulation_steps * ratio))
        
        # Ensure accumulation steps is at least 1
        return max(1, new_steps)
