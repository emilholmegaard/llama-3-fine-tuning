"""
Memory estimation utilities for QLoRA fine-tuning in memory-constrained environments.
Provides functions for estimating memory requirements and suggesting optimizations.
"""

import math
import logging
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import AutoConfig

logger = logging.getLogger(__name__)

def estimate_model_memory_usage(
    model_name_or_path: str,
    bits: int = 16,
    double_quant: bool = False,
    quantization_type: str = "nf4",
    lora_enabled: bool = True,
    lora_r: int = 16,
    lora_target_modules: List[str] = None,
    activation_checkpointing: bool = False,
    sequence_length: int = 2048,
    batch_size: int = 1,
    add_sequence_parallel: bool = False,
    mixed_precision: bool = True
) -> Dict[str, float]:
    """
    Estimate memory usage for a model with given configuration.
    
    Args:
        model_name_or_path: Model identifier or path
        bits: Number of bits for quantization (4, 8, or 16)
        double_quant: Whether to use double quantization
        quantization_type: Quantization type ("nf4" or "fp4")
        lora_enabled: Whether to use LoRA
        lora_r: LoRA rank
        lora_target_modules: Modules to apply LoRA to
        activation_checkpointing: Whether to use activation checkpointing
        sequence_length: Maximum sequence length
        batch_size: Batch size for training
        add_sequence_parallel: Whether to use sequence parallelism
        mixed_precision: Whether to use mixed precision training
        
    Returns:
        Dictionary with memory usage estimates in GB
    """
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name_or_path)
    
    # Get model size parameters
    if hasattr(config, "num_hidden_layers"):
        num_layers = config.num_hidden_layers
    else:
        # Default for LLaMA models
        num_layers = 32
    
    if hasattr(config, "hidden_size"):
        hidden_size = config.hidden_size
    else:
        # Default for LLaMA models
        hidden_size = 4096
    
    if hasattr(config, "num_attention_heads"):
        num_heads = config.num_attention_heads
    else:
        # Default for LLaMA models
        num_heads = 32
    
    # Calculate parameter counts
    model_params = 0
    
    # Embedding layer
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else 32000
    model_params += vocab_size * hidden_size  # Token embeddings
    
    # Transformer layers
    for _ in range(num_layers):
        # Self-attention
        model_params += 4 * hidden_size * hidden_size  # Q, K, V, O projections
        model_params += 2 * hidden_size  # Layer norms
        
        # FFN
        ffn_dim = 4 * hidden_size if not hasattr(config, "intermediate_size") else config.intermediate_size
        model_params += 2 * hidden_size * ffn_dim  # FFN weights
        model_params += ffn_dim + hidden_size  # FFN biases
    
    # Final layer norm
    model_params += hidden_size
    
    # LM head
    model_params += hidden_size * vocab_size
    
    # Calculate bits per parameter
    if bits == 16:
        bytes_per_param = 2
    elif bits == 8:
        bytes_per_param = 1
    else:  # bits == 4
        bytes_per_param = 0.5
        if double_quant:
            # Double quantization further reduces memory slightly
            bytes_per_param *= 0.9
    
    # Calculate model weights memory
    model_weights_memory = model_params * bytes_per_param / (1024 ** 3)  # in GB
    
    # LoRA parameters (much smaller than full model)
    lora_memory = 0
    lora_param_count = 0
    if lora_enabled:
        if lora_target_modules is None:
            # Default target modules for LLMs include QKV attention
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # For each target module
        for _ in lora_target_modules:
            # LoRA adapters use 2 small matrices per target module
            # A and B matrices: hidden_size x r and r x hidden_size
            lora_param_count += 2 * hidden_size * lora_r
        
        # LoRA parameters use full precision (fp32/fp16)
        lora_memory = lora_param_count * (2 if mixed_precision else 4) / (1024 ** 3)  # in GB
    
    # Optimizer states memory
    # For Adam/AdamW, we need 8 bytes per parameter in LoRA (if using LoRA)
    optimizer_memory = 0
    if lora_enabled:
        # Only LoRA parameters are trained
        optimizer_param_count = lora_param_count
    else:
        # Full model parameters are trained
        optimizer_param_count = model_params
    
    # AdamW needs 8 bytes per parameter for optimizer states (2 moments + variances)
    optimizer_memory = optimizer_param_count * 8 / (1024 ** 3)  # in GB
    
    # Activation memory
    # Rough estimation - depends on model architecture and batch size
    activation_size_per_token = hidden_size * 4 * num_layers  # bytes per token
    if activation_checkpointing:
        # Activation checkpointing reduces memory by trading compute
        activation_size_per_token /= math.sqrt(num_layers)
    
    activation_memory = batch_size * sequence_length * activation_size_per_token / (1024 ** 3)  # in GB
    
    # Temporary buffers for gradient accumulation, etc.
    # This is a rough estimate
    buffer_memory = 2.0 if mixed_precision else 4.0  # in GB
    
    # Total estimated memory
    total_memory = model_weights_memory + lora_memory + optimizer_memory + activation_memory + buffer_memory
    
    # Memory savings from specific optimizations
    memory_savings = {
        "activation_checkpointing": 0.0,
        "cpu_offloading": 0.0,
        "mixed_precision": 0.0
    }
    
    if activation_checkpointing:
        memory_savings["activation_checkpointing"] = activation_memory * 0.5
    
    if mixed_precision:
        memory_savings["mixed_precision"] = activation_memory * 0.5
    
    # Return detailed breakdown
    return {
        "model_weights_memory_gb": model_weights_memory,
        "lora_adapter_memory_gb": lora_memory,
        "optimizer_states_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "buffer_memory_gb": buffer_memory,
        "total_estimated_memory_gb": total_memory,
        "memory_savings_gb": memory_savings,
        "model_parameters": model_params,
        "lora_parameters": lora_param_count if lora_enabled else 0
    }

def suggest_memory_optimizations(
    model_name_or_path: str,
    available_memory_gb: float,
    sequence_length: int = 2048,
    batch_size: int = 8,
    target_model_size: Optional[str] = None
) -> Dict[str, Any]:
    """
    Suggest memory optimizations based on model and available GPU memory.
    
    Args:
        model_name_or_path: Model identifier or path
        available_memory_gb: Available GPU memory in GB
        sequence_length: Maximum sequence length
        batch_size: Target batch size
        target_model_size: Optional target model size (e.g., "7B", "13B", "70B")
        
    Returns:
        Dictionary with suggested optimizations
    """
    # Estimate base memory usage (fp16, no special optimizations)
    base_estimate = estimate_model_memory_usage(
        model_name_or_path=model_name_or_path,
        bits=16,
        lora_enabled=False,
        sequence_length=sequence_length,
        batch_size=batch_size,
        mixed_precision=True
    )
    
    total_base_memory = base_estimate["total_estimated_memory_gb"]
    model_params = base_estimate["model_parameters"]
    
    # Determine if we need optimizations
    memory_deficit = total_base_memory - available_memory_gb
    
    # Start with the basic setup
    optimizations = {
        "suggested_setup": {
            "quantization": {
                "enabled": False,
                "bits": 16,
                "double_quant": False,
                "quant_type": None
            },
            "lora": {
                "enabled": False,
                "rank": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "gradient_checkpointing": False,
            "cpu_offloading": False,
            "activation_checkpointing": False,
            "mixed_precision": True,
            "microbatching": False,
            "suggested_batch_size": batch_size,
            "gradient_accumulation": 1,
        },
        "memory_status": {
            "available_memory_gb": available_memory_gb,
            "base_memory_required_gb": total_base_memory,
            "memory_deficit_gb": max(0, memory_deficit),
            "model_parameters": model_params,
            "model_size_gb": model_params * 2 / (1024 ** 3)  # fp16 size in GB
        },
        "explanation": []
    }
    
    # If enough memory, suggest minimal optimizations
    if memory_deficit <= 0:
        optimizations["explanation"].append(
            f"You have sufficient GPU memory ({available_memory_gb:.1f} GB) for this model "
            f"in FP16 precision with batch size {batch_size}."
        )
        optimizations["explanation"].append(
            "For better performance, you can still use LoRA to speed up training."
        )
        optimizations["suggested_setup"]["lora"]["enabled"] = True
        return optimizations
    
    # Start adding optimizations based on deficit
    optimizations["explanation"].append(
        f"Your GPU has {available_memory_gb:.1f} GB available memory, but the model requires "
        f"approximately {total_base_memory:.1f} GB, resulting in a deficit of {memory_deficit:.1f} GB."
    )
    
    # First, enable LoRA (small memory impact, big training benefit)
    optimizations["suggested_setup"]["lora"]["enabled"] = True
    
    # Step 1: Add quantization if memory deficit is significant
    if memory_deficit > 1.0:
        optimizations["suggested_setup"]["quantization"]["enabled"] = True
        optimizations["suggested_setup"]["quantization"]["bits"] = 8
        optimizations["suggested_setup"]["quantization"]["quant_type"] = "int8"
        optimizations["explanation"].append(
            "Enabling 8-bit quantization to reduce model weights memory."
        )
        
        # For higher deficits, use 4-bit quantization
        if memory_deficit > 3.0:
            optimizations["suggested_setup"]["quantization"]["bits"] = 4
            optimizations["suggested_setup"]["quantization"]["quant_type"] = "nf4"
            optimizations["suggested_setup"]["quantization"]["double_quant"] = True
            optimizations["explanation"].append(
                "Upgrading to 4-bit NF4 quantization with double quantization for maximum memory savings."
            )
    
    # Step 2: Enable gradient checkpointing
    if memory_deficit > 2.0:
        optimizations["suggested_setup"]["gradient_checkpointing"] = True
        optimizations["explanation"].append(
            "Enabling gradient checkpointing to reduce activation memory at the cost of some computation speed."
        )
    
    # Step 3: Reduce batch size if still insufficient
    if memory_deficit > 4.0:
        suggested_batch_size = max(1, batch_size // 2)
        optimizations["suggested_setup"]["suggested_batch_size"] = suggested_batch_size
        
        # Add gradient accumulation to maintain effective batch size
        optimizations["suggested_setup"]["gradient_accumulation"] = batch_size // suggested_batch_size
        
        optimizations["explanation"].append(
            f"Reducing batch size from {batch_size} to {suggested_batch_size} and using "
            f"gradient accumulation of {optimizations['suggested_setup']['gradient_accumulation']} steps "
            f"to maintain effective batch size."
        )
    
    # Step 4: Enable CPU offloading for severe memory constraints
    if memory_deficit > 8.0:
        optimizations["suggested_setup"]["cpu_offloading"] = True
        optimizations["explanation"].append(
            "Enabling CPU offloading for optimizer states. This will reduce GPU memory usage "
            "but may significantly slow down training."
        )
    
    # Step 5: For extreme memory constraints, suggest model size reduction
    if memory_deficit > 12.0 and target_model_size:
        current_size = target_model_size
        if current_size == "70B":
            optimizations["explanation"].append(
                "This GPU cannot efficiently train a 70B model. Consider using a smaller model "
                "like 13B or 7B, or switching to a multi-GPU setup."
            )
        elif current_size == "13B":
            optimizations["explanation"].append(
                "This GPU is very constrained for a 13B model. Consider using a 7B model instead "
                "or implementing aggressive CPU offloading and microbatching."
            )
    
    # Step 6: Enable microbatching for extreme memory constraints
    if memory_deficit > 10.0:
        optimizations["suggested_setup"]["microbatching"] = True
        optimizations["explanation"].append(
            "Enabling microbatching to process examples one by one within each batch, "
            "trading speed for memory efficiency."
        )
    
    return optimizations

def find_optimal_batch_size(
    model,
    tokenizer,
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
    from src.utils.memory_utils import clean_memory
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
