"""
Utilities for merging LoRA adapters with base models and stacking multiple LoRA adapters.
"""

import os
import logging
import shutil
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    PeftModel, 
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModelForCausalLM
)

from src.models.llama_wrapper import LlamaWrapper

logger = logging.getLogger(__name__)

class ModelMerger:
    """
    Class for merging LoRA adapters with base models and stacking multiple LoRA adapters.
    """
    
    def __init__(
        self,
        base_model_name_or_path: str,
        precision: str = "fp16",
        device: str = "auto"
    ):
        """
        Initialize the model merger.
        
        Args:
            base_model_name_or_path: Name or path of the base model
            precision: Precision to use ("fp16", "bf16", or "fp32")
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.precision = precision
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            use_fast=True
        )
        
        logger.info(f"ModelMerger initialized with base model: {base_model_name_or_path}")
        
    def _load_base_model(self) -> AutoModelForCausalLM:
        """
        Load the base model with appropriate precision.
        
        Returns:
            Base model
        """
        logger.info(f"Loading base model: {self.base_model_name_or_path}")
        
        # Set torch dtype based on precision
        torch_dtype = None
        if self.precision == "fp16":
            torch_dtype = torch.float16
        elif self.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.precision == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=self.device
        )
        
        return model
    
    def merge_lora_to_base_model(
        self,
        adapter_path: str,
        output_path: str,
        adapter_name: str = "default",
        save_precision: Optional[str] = None,
        quantization_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Merge a LoRA adapter into the base model, creating a standalone model.
        
        Args:
            adapter_path: Path to the LoRA adapter
            output_path: Path to save the merged model
            adapter_name: Name of the adapter
            save_precision: Precision for saving ("fp16", "bf16", "fp32", or None to use loaded precision)
            quantization_config: Quantization configuration for the merged model
            
        Returns:
            Path to the merged model
        """
        # Load base model
        model = self._load_base_model()
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name=adapter_name
        )
        
        # Merge weights
        logger.info("Merging adapter weights into base model")
        model = model.merge_and_unload()
        
        # Determine save precision
        if save_precision:
            save_torch_dtype = None
            if save_precision == "fp16":
                save_torch_dtype = torch.float16
            elif save_precision == "bf16":
                save_torch_dtype = torch.bfloat16
            elif save_precision == "fp32":
                save_torch_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported save precision: {save_precision}")
                
            # Convert model to the desired precision
            if save_torch_dtype:
                model = model.to(save_torch_dtype)
                
        # Apply quantization if specified
        if quantization_config:
            logger.info(f"Applying quantization with config: {quantization_config}")
            # Implementation depends on the quantization method (e.g., bitsandbytes, GPTQ)
            # This is a placeholder for actual quantization logic
            if "bits" in quantization_config:
                bits = quantization_config["bits"]
                logger.info(f"Quantizing to {bits} bits")
                # Apply quantization
                # model = quantize_model(model, bits)
        
        # Save merged model and tokenizer
        logger.info(f"Saving merged model to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save quantization config if provided
        if quantization_config:
            import json
            with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
                json.dump(quantization_config, f, indent=2)
        
        return output_path
    
    def stack_adapters(
        self,
        adapter_paths: List[str],
        output_path: str,
        adapter_weights: Optional[List[float]] = None
    ) -> str:
        """
        Stack multiple LoRA adapters, creating a new combined adapter.
        
        Args:
            adapter_paths: List of paths to LoRA adapters
            output_path: Path to save the stacked adapter
            adapter_weights: Optional list of weights for each adapter (must sum to 1.0)
            
        Returns:
            Path to the stacked adapter
        """
        if len(adapter_paths) < 2:
            raise ValueError("At least two adapters required for stacking")
        
        # Validate adapter weights if provided
        if adapter_weights is not None:
            if len(adapter_weights) != len(adapter_paths):
                raise ValueError("Number of adapter weights must match number of adapters")
            if abs(sum(adapter_weights) - 1.0) > 1e-6:
                raise ValueError("Adapter weights must sum to 1.0")
        else:
            # Equal weighting by default
            adapter_weights = [1.0 / len(adapter_paths) for _ in adapter_paths]
        
        logger.info(f"Stacking {len(adapter_paths)} adapters with weights {adapter_weights}")
        
        # Load base model
        model = self._load_base_model()
        
        # Load the first adapter to get its configuration
        logger.info(f"Loading first adapter from {adapter_paths[0]}")
        first_adapter_config = PeftConfig.from_pretrained(adapter_paths[0])
        
        # We need to create a new LoRA configuration for the stacked model
        stacked_lora_config = LoraConfig(
            r=first_adapter_config.r,
            lora_alpha=first_adapter_config.lora_alpha,
            target_modules=first_adapter_config.target_modules,
            lora_dropout=first_adapter_config.lora_dropout,
            bias=first_adapter_config.bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        # Create a new PEFT model with the stacked configuration
        stacked_model = get_peft_model(model, stacked_lora_config)
        
        # Get adapter weights from all adapters
        adapters_weights_dict = {}
        
        for i, adapter_path in enumerate(adapter_paths):
            weight = adapter_weights[i]
            logger.info(f"Processing adapter {i+1}/{len(adapter_paths)} with weight {weight}")
            
            # Load adapter state dict
            adapter_state_dict = {}
            adapter_loaded = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu")
            
            # Process each weight
            for key, value in adapter_loaded.items():
                # Skip non-weight keys
                if not key.endswith(".weight") and not key.endswith(".bias"):
                    continue
                
                # Apply weight to this adapter's parameters
                adapter_state_dict[key] = value * weight
                
                # Add to accumulated dictionary if not already present
                if key not in adapters_weights_dict:
                    adapters_weights_dict[key] = adapter_state_dict[key]
                else:
                    # Add weighted values
                    adapters_weights_dict[key] += adapter_state_dict[key]
                
        # Load combined weights into stacked model
        stacked_state_dict = stacked_model.state_dict()
        for key, value in adapters_weights_dict.items():
            if key in stacked_state_dict:
                stacked_state_dict[key] = value
                
        stacked_model.load_state_dict(stacked_state_dict)
                
        # Save stacked adapter
        logger.info(f"Saving stacked adapter to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        stacked_model.save_pretrained(output_path)
        
        # Save adapter config
        config_dict = {
            "base_model": self.base_model_name_or_path,
            "stacked_adapters": [os.path.basename(path) for path in adapter_paths],
            "adapter_weights": adapter_weights
        }
        
        import json
        with open(os.path.join(output_path, "stacked_adapter_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        return output_path
    
    def quantize_merged_model(
        self,
        model_path: str,
        output_path: str,
        quantization_config: Dict[str, Any]
    ) -> str:
        """
        Quantize a merged model to reduce its size.
        
        Args:
            model_path: Path to the model to quantize
            output_path: Path to save the quantized model
            quantization_config: Quantization configuration
            
        Returns:
            Path to the quantized model
        """
        logger.info(f"Quantizing model from {model_path} to {output_path}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Load model
        from transformers import BitsAndBytesConfig
        
        # Create quantization config
        bits = quantization_config.get("bits", 4)
        quantization_type = quantization_config.get("quant_type", "nf4")
        
        # Configure BitsAndBytes settings
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            bnb_4bit_compute_dtype=torch.float16,  # Can be float16 or bfloat16
            bnb_4bit_quant_type=quantization_type,
            bnb_4bit_use_double_quant=quantization_config.get("double_quant", True)
        )
        
        # Load and quantize model
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        # Save the quantized model
        quantized_model.save_pretrained(output_path)
        
        # Also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(output_path)
        
        # Save quantization config
        with open(os.path.join(output_path, "quantization_config.json"), "w") as f:
            import json
            json.dump(quantization_config, f, indent=2)
        
        logger.info(f"Quantized model saved to {output_path}")
        return output_path
