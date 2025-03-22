"""
Memory-efficient trainer implementation for QLoRA fine-tuning.
Integrates advanced memory optimizations for training in constrained environments.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl, 
    PreTrainedModel, 
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from src.utils.memory_utils import clean_memory, print_gpu_memory_summary
from src.utils.memory_tracker import MemoryTracker
from src.utils.memory_estimation import suggest_memory_optimizations
from src.utils.cpu_offload import enable_selective_activation_checkpointing
from src.utils.batch_optimizer import find_optimal_batch_size, DynamicGradientAccumulation, MicroBatchingDataloader
from src.models.qlora_optimizer import QLoRAOptimizer

logger = logging.getLogger(__name__)

class MemoryEfficientTrainer:
    """
    Memory-efficient trainer for QLoRA fine-tuning.
    Integrates all memory optimization techniques.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        target_modules: Optional[List[str]] = None,
        bits: int = 4,
        double_quant: bool = True,
        quant_type: str = "nf4",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        memory_profiling: bool = True,
        optimize_for_memory: bool = True,
        auto_find_batch_size: bool = True,
        auto_find_parameters: bool = True,
        log_dir: Optional[str] = None,
        device_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize memory-efficient trainer.
        
        Args:
            model_name_or_path: Model name or path
            output_dir: Output directory for checkpoints and logs
            target_modules: Target modules for LoRA (None for default)
            bits: Quantization bits (4 or 8)
            double_quant: Whether to use double quantization
            quant_type: Quantization type ('nf4' or 'fp4')
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            memory_profiling: Whether to enable memory profiling
            optimize_for_memory: Whether to apply aggressive memory optimizations
            auto_find_batch_size: Whether to automatically find optimal batch size
            auto_find_parameters: Whether to automatically find optimal parameters
            log_dir: Directory for memory logs (None for default)
            device_map: Device mapping for model (None for auto)
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.target_modules = target_modules
        self.bits = bits
        self.double_quant = double_quant
        self.quant_type = quant_type
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.memory_profiling = memory_profiling
        self.optimize_for_memory = optimize_for_memory
        self.auto_find_batch_size = auto_find_batch_size
        self.auto_find_parameters = auto_find_parameters
        self.log_dir = log_dir or os.path.join(output_dir, "memory_logs")
        self.device_map = device_map
        
        # Calculated values
        self.optimal_batch_size = None
        self.optimal_gradient_accumulation = None
        self.use_cpu_offloading = False
        self.use_gradient_checkpointing = True
        self.use_activation_checkpointing = False
        
        # Create components
        self.qlora_optimizer = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.memory_tracker = None
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def optimize_for_available_memory(self) -> Dict[str, Any]:
        """
        Optimize hyperparameters for available memory.
        
        Returns:
            Dictionary with optimized hyperparameters
        """
        # Get available memory
        available_memory_gb = 0
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory_gb = available_memory / (1024 ** 3)
            
            # Leave some headroom
            available_memory_gb *= 0.95
        
        # Get optimization suggestions
        suggestions = suggest_memory_optimizations(
            model_name_or_path=self.model_name_or_path,
            available_memory_gb=available_memory_gb,
            sequence_length=2048,  # Default sequence length
            batch_size=8,  # Default batch size
            target_model_size=None  # Determine from model name
        )
        
        # Update parameters based on suggestions
        setup = suggestions["suggested_setup"]
        
        # Update quantization settings
        if setup["quantization"]["enabled"]:
            self.bits = setup["quantization"]["bits"]
            self.double_quant = setup["quantization"]["double_quant"]
            self.quant_type = setup["quantization"]["quant_type"]
        
        # Update LoRA settings
        if setup["lora"]["enabled"]:
            self.lora_r = setup["lora"]["rank"]
            if setup["lora"]["target_modules"]:
                self.target_modules = setup["lora"]["target_modules"]
        
        # Update memory optimization flags
        self.use_gradient_checkpointing = setup["gradient_checkpointing"]
        self.use_cpu_offloading = setup["cpu_offloading"]
        self.use_activation_checkpointing = setup["activation_checkpointing"]
        
        # Update batch size and gradient accumulation
        if setup["suggested_batch_size"]:
            self.optimal_batch_size = setup["suggested_batch_size"]
        if setup["gradient_accumulation"] > 1:
            self.optimal_gradient_accumulation = setup["gradient_accumulation"]
        
        # Log the optimizations
        logger.info(f"Memory optimizations applied based on {available_memory_gb:.2f}GB available memory:")
        for explanation in suggestions["explanation"]:
            logger.info(f"- {explanation}")
        
        return setup
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load and optimize model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Optimize parameters based on available memory if requested
        if self.auto_find_parameters and self.optimize_for_memory:
            self.optimize_for_available_memory()
        
        # Initialize QLoRA optimizer
        self.qlora_optimizer = QLoRAOptimizer(
            bits=self.bits,
            double_quant=self.double_quant,
            quant_type=self.quant_type,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            memory_profiling=self.memory_profiling,
            cpu_offloading=self.use_cpu_offloading,
            gradient_checkpointing=self.use_gradient_checkpointing,
            log_dir=self.log_dir
        )
        
        # Load tokenizer
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {self.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load quantized model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            bnb_4bit_use_double_quant=self.double_quant,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model
        logger.info(f"Loading model from {self.model_name_or_path} with {self.bits}-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=torch.float16,
        )
        
        # Set components in QLoRA optimizer
        self.qlora_optimizer.set_tokenizer(tokenizer)
        self.qlora_optimizer.set_model(model)
        
        # Apply additional memory optimizations
        if self.use_activation_checkpointing:
            # Apply selective activation checkpointing
            model = enable_selective_activation_checkpointing(model)
        
        # Store references
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def prepare_training_data(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
        text_column: str = "text",
        max_seq_length: int = 2048,
        num_workers: int = 4,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare datasets for training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            text_column: Name of text column in datasets
            max_seq_length: Maximum sequence length
            num_workers: Number of preprocessing workers
            
        Returns:
            Prepared datasets
        """
        logger.info("Preparing training data")
        
        def tokenize_function(examples):
            # Tokenize texts
            return self.tokenizer(
                examples[text_column],
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None
            )
        
        # Process all splits in the dataset
        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=[col for col in train_dataset.column_names if col != text_column],
            desc="Tokenizing training data",
        )
        
        eval_tokenized = None
        if eval_dataset is not None:
            eval_tokenized = eval_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=num_workers,
                remove_columns=[col for col in eval_dataset.column_names if col != text_column],
                desc="Tokenizing validation data",
            )
        
        return train_tokenized, eval_tokenized
    
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        warmup_ratio: float = 0.03,
        logging_steps: int = 100,
        save_steps: int = 500,
        save_total_limit: int = 3,
        **kwargs
    ) -> Trainer:
        """
        Create memory-efficient trainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (None for auto)
            gradient_accumulation_steps: Gradient accumulation steps (None for auto)
            warmup_ratio: Warmup ratio
            logging_steps: Logging steps
            save_steps: Save steps
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional arguments to pass to Trainer
            
        Returns:
            Trainer instance
        """
        # Find optimal batch size if requested
        if batch_size is None and self.auto_find_batch_size:
            if self.optimal_batch_size is None:
                logger.info("Finding optimal batch size")
                self.optimal_batch_size = find_optimal_batch_size(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    start_batch_size=8,
                    min_batch_size=1,
                    max_sequence_length=2048
                )
            batch_size = self.optimal_batch_size
        elif batch_size is None:
            # Default batch size
            batch_size = 4
        
        # Calculate gradient accumulation steps
        if gradient_accumulation_steps is None:
            if self.optimal_gradient_accumulation is None:
                # Default to 4 if not calculated
                gradient_accumulation_steps = 4
            else:
                gradient_accumulation_steps = self.optimal_gradient_accumulation
        
        # Create dynamic gradient accumulation
        dynamic_accumulation = DynamicGradientAccumulation(
            base_batch_size=batch_size,
            base_accumulation_steps=gradient_accumulation_steps,
            min_batch_size=1,
            max_accumulation_steps=32
        )
        
        # Create training arguments
        training_args = self.qlora_optimizer.create_training_args(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            **kwargs
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = self.qlora_optimizer.create_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            data_collator=data_collator,
            **kwargs
        )
        
        self.trainer = trainer
        return trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Run the training process.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer must be created before calling train")
        
        # Clean memory before training
        clean_memory()
        
        logger.info("Starting training with memory-efficient QLoRA")
        if self.memory_profiling:
            print_gpu_memory_summary()
        
        # Run training
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the model and trainer state
        self.trainer.save_model(self.output_dir)
        self.trainer.save_state()
        
        # Save memory log if profiling was enabled
        if hasattr(self.qlora_optimizer, "memory_tracker") and self.qlora_optimizer.memory_tracker:
            memory_log_path = self.qlora_optimizer.memory_tracker.save_log("memory_log_final.json")
            logger.info(f"Memory log saved to {memory_log_path}")
        
        # Log metrics
        metrics = train_result.metrics
        
        # Log metrics
        max_train_samples = len(self.trainer.train_dataset)
        metrics["train_samples"] = min(max_train_samples, metrics.get("train_samples", 0))
        
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer must be created before calling evaluate")
        
        logger.info("Running evaluation")
        
        metrics = self.trainer.evaluate()
        max_eval_samples = len(self.trainer.eval_dataset) if self.trainer.eval_dataset else 0
        metrics["eval_samples"] = max_eval_samples
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def save_hyperparameters(self, path: Optional[str] = None) -> None:
        """
        Save hyperparameters to file.
        
        Args:
            path: Path to save hyperparameters to (None for default)
        """
        if path is None:
            path = os.path.join(self.output_dir, "hyperparameters.json")
        
        hyperparameters = {
            "model_name_or_path": self.model_name_or_path,
            "bits": self.bits,
            "double_quant": self.double_quant,
            "quant_type": self.quant_type,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_cpu_offloading": self.use_cpu_offloading,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_activation_checkpointing": self.use_activation_checkpointing,
            "optimal_batch_size": self.optimal_batch_size,
            "optimal_gradient_accumulation": self.optimal_gradient_accumulation,
        }
        
        with open(path, "w") as f:
            json.dump(hyperparameters, f, indent=2)
        
        logger.info(f"Hyperparameters saved to {path}")
