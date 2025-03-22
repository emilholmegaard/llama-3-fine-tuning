"""
QLoRA optimization implementation for Llama fine-tuning.
Provides specialized optimizations for training in memory-constrained environments.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

from src.utils.memory_utils import clean_memory, get_gpu_memory_info, print_gpu_memory_summary
from src.utils.cpu_offload import CPUOffloadOptimizer, enable_selective_activation_checkpointing
from src.utils.memory_tracker import MemoryTracker
from src.utils.memory_estimation import estimate_model_memory_usage, find_optimal_batch_size

logger = logging.getLogger(__name__)

class MemoryTrackingCallback(TrainerCallback):
    """
    Callback for tracking memory usage during training.
    """
    
    def __init__(self, memory_tracker: MemoryTracker, log_interval: int = 10):
        """
        Initialize memory tracking callback.
        
        Args:
            memory_tracker: MemoryTracker instance
            log_interval: How often to log memory statistics (in steps)
        """
        self.memory_tracker = memory_tracker
        self.log_interval = log_interval
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """
        Track memory at the end of each training step.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments
            
        Returns:
            Trainer control
        """
        self.memory_tracker.update(step=state.global_step)
        return control
    
    def on_log(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """
        Save memory log at each logging interval.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments
            
        Returns:
            Trainer control
        """
        # Force log at each logging step
        self.memory_tracker.update(step=state.global_step, force_log=True)
        
        # Save log every 10 logging steps
        if state.global_step % (args.logging_steps * 10) == 0:
            self.memory_tracker.save_log(filename=f"memory_log_step_{state.global_step}.json")
        
        return control
    
    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """
        Save final memory log at the end of training.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional arguments
            
        Returns:
            Trainer control
        """
        # Save final memory log
        self.memory_tracker.save_log(filename="memory_log_final.json")
        return control


class QLoRAOptimizer:
    """
    Advanced QLoRA optimization for memory-constrained Llama fine-tuning.
    """
    
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        bits: int = 4,
        double_quant: bool = True,
        quant_type: str = "nf4",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        memory_profiling: bool = True,
        cpu_offloading: bool = False,
        gradient_checkpointing: bool = True,
        log_dir: str = "memory_logs",
        offload_dir: Optional[str] = None,
    ):
        """
        Initialize QLoRA optimizer.
        
        Args:
            model: Model to optimize (can also be set later)
            tokenizer: Tokenizer for the model (can also be set later)
            bits: Quantization bits (4 or 8)
            double_quant: Whether to use double quantization
            quant_type: Quantization type ('nf4' or 'fp4')
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            target_modules: List of module types to target for LoRA
            memory_profiling: Whether to enable memory profiling
            cpu_offloading: Whether to offload optimizer states to CPU
            gradient_checkpointing: Whether to use gradient checkpointing
            log_dir: Directory to save memory logs
            offload_dir: Directory for CPU offloading (if None, uses temp)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.memory_profiling = memory_profiling
        self.cpu_offloading = cpu_offloading
        self.gradient_checkpointing = gradient_checkpointing
        self.log_dir = log_dir
        self.offload_dir = offload_dir or os.path.join(os.getcwd(), ".temp_offload")
        
        # Set default target modules if not provided
        if self.target_modules is None:
            # Default target modules for LLaMA model
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Initialize memory tracker if profiling is enabled
        self.memory_tracker = MemoryTracker(log_dir=log_dir) if memory_profiling else None
        
        # Initialize the model if both model and tokenizer are provided
        if self.model is not None and self.tokenizer is not None:
            self.prepare_model()
        
        # Make sure offload directory exists
        if self.cpu_offloading and self.offload_dir:
            os.makedirs(self.offload_dir, exist_ok=True)
    
    def estimate_memory_usage(
        self, 
        model_name_or_path: str, 
        sequence_length: int = 2048, 
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Estimate memory usage with current optimization settings.
        
        Args:
            model_name_or_path: Model name or path
            sequence_length: Sequence length for training
            batch_size: Batch size for training
            
        Returns:
            Dictionary with memory usage estimates
        """
        return estimate_model_memory_usage(
            model_name_or_path=model_name_or_path,
            bits=self.bits,
            double_quant=self.double_quant,
            quantization_type=self.quant_type,
            lora_enabled=True,
            lora_r=self.lora_r,
            lora_target_modules=self.target_modules,
            activation_checkpointing=self.gradient_checkpointing,
            sequence_length=sequence_length,
            batch_size=batch_size,
            mixed_precision=True
        )
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with QLoRA optimizations.
        
        Returns:
            Optimized model
        """
        if self.model is None:
            raise ValueError("Model must be set before calling prepare_model")
        
        # Print initial memory usage
        if self.memory_profiling:
            print_gpu_memory_summary()
            self.memory_tracker.start_tracking()
            self.memory_tracker.update(step=0, force_log=True)
        
        logger.info(f"Preparing model with {self.bits}-bit quantization and LoRA (r={self.lora_r})")
        
        # Clean memory before starting
        clean_memory()
        
        # 1. Apply quantization
        from transformers import BitsAndBytesConfig
        
        # Set up quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            bnb_4bit_use_double_quant=self.double_quant,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # 2. Apply k-bit training preparation
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.gradient_checkpointing
        )
        
        if self.memory_profiling:
            self.memory_tracker.update(step=1, force_log=True)
            logger.info("Memory usage after k-bit training preparation")
        
        # 3. Apply LoRA
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
            fan_in_fan_out=False
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        if self.memory_profiling:
            self.memory_tracker.update(step=2, force_log=True)
            logger.info("Memory usage after applying LoRA adapter")
        
        # 4. Apply gradient checkpointing if requested
        if self.gradient_checkpointing:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        if self.memory_profiling:
            self.memory_tracker.update(step=3, force_log=True)
            memory_summary = self.memory_tracker.get_summary()
            logger.info(f"Memory usage after model preparation: {memory_summary['peak_memory_mb']:.2f}MB peak")
        
        return self.model
    
    def create_optimizer(
        self, 
        learning_rate: float = 2e-5, 
        weight_decay: float = 0.01, 
        optimizer_type: str = "adamw",
        beta1: float = 0.9, 
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with appropriate optimizations.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            optimizer_type: Optimizer type ('adamw', 'adamw_8bit', or 'adafactor')
            beta1: Beta1 parameter for Adam-based optimizers
            beta2: Beta2 parameter for Adam-based optimizers
            epsilon: Epsilon parameter for Adam-based optimizers
            
        Returns:
            Optimized optimizer
        """
        if self.model is None:
            raise ValueError("Model must be set and prepared before creating optimizer")
        
        # Get trainable parameters
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=epsilon,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw_8bit':
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=epsilon,
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                    betas=(beta1, beta2),
                    eps=epsilon,
                    weight_decay=weight_decay
                )
        elif optimizer_type.lower() == 'adafactor':
            try:
                from transformers.optimization import Adafactor
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Adafactor not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=learning_rate
                )
        else:
            logger.warning(f"Unknown optimizer type {optimizer_type}, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Apply CPU offloading if requested
        if self.cpu_offloading:
            logger.info("Applying CPU offloading to optimizer states")
            optimizer = CPUOffloadOptimizer(optimizer)
        
        return optimizer
    
    def find_optimal_batch_size(
        self, 
        start_batch_size: int = 16, 
        min_batch_size: int = 1, 
        max_sequence_length: int = 2048
    ) -> int:
        """
        Find optimal batch size for the model with current optimizations.
        
        Args:
            start_batch_size: Starting batch size to try
            min_batch_size: Minimum acceptable batch size
            max_sequence_length: Maximum sequence length
            
        Returns:
            Optimal batch size
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before finding optimal batch size")
        
        return find_optimal_batch_size(
            model=self.model,
            tokenizer=self.tokenizer,
            start_batch_size=start_batch_size,
            max_sequence_length=max_sequence_length,
            min_batch_size=min_batch_size
        )
    
    def create_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 100,
        save_steps: int = 500,
        save_total_limit: int = 3,
        **kwargs
    ) -> TrainingArguments:
        """
        Create optimized training arguments.
        
        Args:
            output_dir: Output directory
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_ratio: Warmup ratio
            logging_steps: Logging steps
            save_steps: Save steps
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional arguments to pass to TrainingArguments
            
        Returns:
            TrainingArguments instance
        """
        # Prepare arguments with optimized settings
        args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "report_to": ["tensorboard"],
            
            # Memory optimizations
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": torch.cuda.is_bf16_supported(),  # Use bf16 if supported
            "fp16": not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),  # Use fp16 as fallback
            
            # LR scheduling
            "lr_scheduler_type": "cosine",
            
            # Evaluation settings
            "evaluation_strategy": "steps",
            "eval_steps": save_steps,
            
            # Weight decay
            "weight_decay": 0.01,
            
            # Gradient clipping
            "max_grad_norm": 1.0,
        }
        
        # Update with additional arguments
        args.update(kwargs)
        
        # Create training arguments
        training_args = TrainingArguments(**args)
        
        return training_args
    
    def create_trainer(
        self,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        data_collator=None,
        **kwargs
    ) -> Trainer:
        """
        Create optimized trainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            data_collator: Data collator
            **kwargs: Additional arguments to pass to Trainer
            
        Returns:
            Optimized Trainer instance
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before creating trainer")
        
        # Create default training args if not provided
        if training_args is None:
            training_args = self.create_training_args(output_dir="output")
        
        # Create default data collator if not provided
        if data_collator is None:
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        
        # Set up callbacks
        callbacks = kwargs.pop("callbacks", [])
        
        # Add memory tracking callback if enabled
        if self.memory_profiling and self.memory_tracker:
            memory_callback = MemoryTrackingCallback(
                memory_tracker=self.memory_tracker,
                log_interval=training_args.logging_steps
            )
            callbacks.append(memory_callback)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            **kwargs
        )
        
        return trainer
    
    def set_model(self, model: PreTrainedModel) -> None:
        """
        Set model to optimize.
        
        Args:
            model: Model to optimize
        """
        self.model = model
        
        # Prepare model if tokenizer is also set
        if self.tokenizer is not None:
            self.prepare_model()
    
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Set tokenizer to use with model.
        
        Args:
            tokenizer: Tokenizer to use
        """
        self.tokenizer = tokenizer
        
        # Prepare model if model is also set
        if self.model is not None:
            self.prepare_model()
