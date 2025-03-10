"""
Llama model wrapper for fine-tuning and inference.
"""

import os
import logging
from typing import Dict, Any, Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

class LlamaWrapper:
    """
    Wrapper class for Llama 3.3 models that handles loading, saving, and inference.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        **kwargs
    ):
        """
        Initialize the Llama wrapper.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
            tokenizer_name_or_path: Path to tokenizer or identifier (defaults to model_name_or_path)
            quantization_config: Configuration for quantization (for QLoRA)
            device_map: Device mapping strategy for model loading
            **kwargs: Additional arguments to pass to AutoModelForCausalLM.from_pretrained
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.device_map = device_map
        
        # Setup quantization config if provided
        if quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_config.get("bits", 4) == 4,
                load_in_8bit=quantization_config.get("bits", 4) == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=quantization_config.get("double_quant", True),
                bnb_4bit_quant_type=quantization_config.get("quant_type", "nf4")
            )
            kwargs["quantization_config"] = bnb_config
        
        logger.info(f"Loading model from {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16,  # Use fp16 by default for efficiency
            **kwargs
        )
        
        logger.info(f"Loading tokenizer from {self.tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            use_fast=True
        )
        
        # Ensure the tokenizer has padding token for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    def save(self, output_dir: str, save_adapter_only: bool = False) -> None:
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory where model/adapter will be saved
            save_adapter_only: If True, only save the adapter weights (for LoRA models)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # If using PEFT adapter and save_adapter_only is True, just save the adapter
        if hasattr(self.model, "save_pretrained") and not save_adapter_only:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Full model and tokenizer saved to {output_dir}")
        elif hasattr(self.model, "save_pretrained") and save_adapter_only:
            # For adapter-only saving, typically with PEFT models
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Model adapter and tokenizer saved to {output_dir}")
        else:
            logger.warning(f"Model does not support save_pretrained. Skipping save to {output_dir}")
    
    def generate(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a list of prompts.
        
        Args:
            prompts: List of text prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional arguments to pass to model.generate
            
        Returns:
            List of generated texts
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode the generated text
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        results = []
        for i, text in enumerate(generated_texts):
            # Get just the generated part by decoding just the generated tokens
            generated_part = self.tokenizer.decode(
                outputs[i][prompt_lengths[i]:], 
                skip_special_tokens=True
            )
            results.append(generated_part)
        
        return results
