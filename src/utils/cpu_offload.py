"""
CPU offloading utilities for QLoRA fine-tuning in memory-constrained environments.
Provides CPU offloading for models and optimizers to reduce GPU memory usage.
"""

import os
import logging
from typing import Dict, Optional, Any, Tuple

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

class CPUOffloadOptimizer:
    """
    Wrapper for optimizer that offloads states to CPU to save GPU memory.
    """
    
    def __init__(self, optimizer, device="cuda"):
        """
        Initialize CPU offloaded optimizer.
        
        Args:
            optimizer: Base optimizer
            device: Device for active gradients
        """
        self.optimizer = optimizer
        self.device = device
        self.state_device = "cpu"
        
        # Save original functions
        self._orig_step = optimizer.step
        self._orig_zero_grad = optimizer.zero_grad
        
        # Override step
        def _step(closure=None):
            # Move states to device for the update
            self._move_states_to_device(self.device)
            
            # Run original step
            loss = self._orig_step(closure)
            
            # Move states back to CPU
            self._move_states_to_device(self.state_device)
            
            return loss
        
        # Replace optimizer step
        self.optimizer.step = _step
        
        # Move states to CPU initially
        self._move_states_to_device(self.state_device)
        
        logger.info(f"Initialized CPU offloaded optimizer with {len(optimizer.param_groups)} parameter groups")
    
    def _move_states_to_device(self, device: str) -> None:
        """
        Move optimizer states to specified device.
        
        Args:
            device: Target device to move states to
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Access optimizer state for this parameter
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        
                        # Move each state tensor to the device
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                state[key] = value.to(device)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary.
        
        Returns:
            Optimizer state dictionary
        """
        # Move to device for creating state dict
        self._move_states_to_device(self.device)
        state_dict = self.optimizer.state_dict()
        self._move_states_to_device(self.state_device)
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state dictionary.
        
        Args:
            state_dict: Optimizer state dictionary
        """
        # Move to device before loading
        self._move_states_to_device(self.device)
        self.optimizer.load_state_dict(state_dict)
        self._move_states_to_device(self.state_device)
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zero gradients.
        
        Args:
            set_to_none: Whether to set gradients to None
        """
        self._orig_zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """Get parameter groups."""
        return self.optimizer.param_groups


def optimize_model_for_inference(
    model: PreTrainedModel,
    tokenizer,
    device: Optional[str] = None,
    max_memory: Optional[Dict[str, str]] = None,
    offload_folder: Optional[str] = None,
    cpu_offload: bool = False
) -> PreTrainedModel:
    """
    Apply optimizations to a model for memory-efficient inference.
    
    Args:
        model: Model to optimize
        tokenizer: Tokenizer to use with the model
        device: Device to place the model on ("cpu", "cuda", etc.)
        max_memory: Dictionary mapping device to maximum memory
        offload_folder: Directory to offload weights to
        cpu_offload: Whether to offload weights to CPU
        
    Returns:
        Optimized model
    """
    # Apply optimizations
    if device:
        model = model.to(device)
    
    # Apply CPU offloading if specified
    if cpu_offload:
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
        
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            
            # Get device map for efficient distribution
            if max_memory is None:
                max_memory = {}
                if torch.cuda.is_available():
                    # Leave some headroom for the system
                    free_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) * 0.9
                    max_memory["cuda:0"] = f"{free_memory:.2f}GiB"
                max_memory["cpu"] = "32GiB"  # CPU can use more memory
            
            # Infer optimal device map
            device_map = infer_auto_device_map(
                model, 
                max_memory=max_memory,
                no_split_module_classes=[]  # Model-specific non-splittable modules
            )
            
            # Apply device map for efficient memory usage
            model = dispatch_model(model, device_map=device_map, offload_dir=offload_folder)
            logger.info(f"Applied CPU offloading with device map: {device_map}")
        except ImportError:
            logger.warning("accelerate library not available, skipping CPU offloading")
    
    # Apply model optimizations
    if hasattr(model, "config"):
        if hasattr(model.config, "use_cache"):
            # Enable KV caching for faster inference
            model.config.use_cache = True
    
    # Return optimized model
    return model


def enable_selective_activation_checkpointing(model: PreTrainedModel, layer_ids: Optional[list] = None) -> PreTrainedModel:
    """
    Enables activation checkpointing on specific transformer layers to save memory.
    
    Args:
        model: Hugging Face transformer model
        layer_ids: List of layer indices to apply checkpointing to (None for all)
    
    Returns:
        Model with activation checkpointing enabled
    """
    # Check model type
    if not hasattr(model, "enable_input_require_grads"):
        logger.warning("Model does not support enable_input_require_grads, cannot apply activation checkpointing")
        return model
    
    # Enable input gradients for checkpointing
    model.enable_input_require_grads()
    
    # Get all transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        logger.warning("Could not find transformer layers, skipping activation checkpointing")
        return model
    
    # If layer_ids not provided, checkpoint all layers
    if layer_ids is None:
        layer_ids = list(range(len(layers)))
    
    # Apply checkpointing
    from torch.utils.checkpoint import checkpoint
    
    # Get number of layers
    num_layers = len(layers)
    
    # Define checkpointing function
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    # Apply checkpointing to selected layers
    total_applied = 0
    for idx in layer_ids:
        if idx < 0 or idx >= num_layers:
            logger.warning(f"Layer index {idx} out of range, skipping")
            continue
            
        layer = layers[idx]
        
        # Save original forward
        orig_forward = layer.forward
        
        # Apply checkpointing
        def get_checkpointed_forward(original_forward):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(create_custom_forward(original_forward), *args, **kwargs)
            return checkpointed_forward
        
        layer.forward = get_checkpointed_forward(orig_forward)
        total_applied += 1
    
    logger.info(f"Applied activation checkpointing to {total_applied} layers")
    return model
