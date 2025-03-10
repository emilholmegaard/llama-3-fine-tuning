"""
Fine-tuning utilities for Llama models using LoRA and QLoRA.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, List, Union

import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

from src.models.llama_wrapper import LlamaWrapper

logger = logging.getLogger(__name__)

class LlamaFineTuner:
    """
    Fine-tuning class for Llama models using PEFT methods (LoRA and QLoRA).
    """
    
    def __init__(
        self,
        config_path: str,
        local_rank: int = -1
    ):
        """
        Initialize the fine-tuner from a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            local_rank: Local rank for distributed training
        """
        self.local_rank = local_rank
        self.load_config(config_path)
        self.setup_model()
        
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set key configurations
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        self.lora_config = self.config.get('lora', {})
        self.qlora_config = self.config.get('qlora', {})
        self.evaluation_config = self.config.get('evaluation', {})
        self.data_config = self.config.get('data', {})
        self.tracking_config = self.config.get('tracking', {})
        
        # Set important paths
        self.base_model_name = self.model_config.get('base_model', 'meta-llama/Llama-3.3-8B')
        self.output_dir = self.model_config.get('output_dir', 'data/models/finetuned-model/')
        
        logger.info(f"Configuration loaded from {config_path}")
        
    def setup_model(self) -> None:
        """Set up the model with LoRA or QLoRA based on configuration."""
        # Determine if using QLoRA
        use_qlora = self.qlora_config.get('use_qlora', False)
        use_lora = self.lora_config.get('use_lora', True)
        
        if not (use_lora or use_qlora):
            raise ValueError("Either LoRA or QLoRA must be enabled for parameter-efficient fine-tuning")
            
        quantization_config = None
        if use_qlora:
            logger.info("Setting up model with QLoRA")
            quantization_config = {
                'bits': self.qlora_config.get('bits', 4),
                'double_quant': self.qlora_config.get('double_quant', True),
                'quant_type': self.qlora_config.get('quant_type', 'nf4')
            }
            
        # Initialize the base model with appropriate quantization
        self.llama_wrapper = LlamaWrapper(
            model_name_or_path=self.base_model_name,
            quantization_config=quantization_config
        )
        
        # Apply LoRA adapter
        if use_qlora or use_lora:
            self._apply_lora_adapter()
        
        # Set eval to False for training
        self.llama_wrapper.model.train()
        logger.info(f"Model setup complete: {self.base_model_name}")
    
    def _apply_lora_adapter(self) -> None:
        """Apply LoRA adapter to the model."""
        # Prepare model for k-bit training if using QLoRA
        if self.qlora_config.get('use_qlora', False):
            self.llama_wrapper.model = prepare_model_for_kbit_training(
                self.llama_wrapper.model,
                use_gradient_checkpointing=True
            )
            
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('alpha', 32),
            lora_dropout=self.lora_config.get('dropout', 0.05),
            bias=self.lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_config.get(
                'target_modules', 
                ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        )
        
        # Apply LoRA adapter
        self.llama_wrapper.model = get_peft_model(self.llama_wrapper.model, lora_config)
        logger.info(f"Applied LoRA adapter with rank {lora_config.r}")
        
        # Print trainable parameters info
        self.llama_wrapper.model.print_trainable_parameters()
        
    def prepare_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict],
        text_column: str = "text"
    ) -> Union[Dataset, DatasetDict]:
        """
        Tokenize and prepare dataset for training.
        
        Args:
            dataset: Dataset or DatasetDict to prepare
            text_column: Name of the column containing text data
            
        Returns:
            Processed dataset ready for training
        """
        max_seq_length = self.data_config.get('max_seq_length', 2048)
        
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.llama_wrapper.tokenizer(
                examples[text_column],
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None
            )
            return tokenized
        
        # Process all splits in the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.data_config.get('preprocessing_num_workers', 4),
            remove_columns=[col for col in dataset.column_names if col != text_column],
            desc="Tokenizing dataset",
        )
        
        logger.info(f"Dataset prepared with max sequence length {max_seq_length}")
        return tokenized_dataset
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        """
        Create a Trainer instance for fine-tuning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Trainer instance
        """
        # Setup data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.llama_wrapper.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            learning_rate=self.training_config.get('learning_rate', 2e-5),
            per_device_train_batch_size=self.training_config.get('batch_size', 8),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
            num_train_epochs=self.training_config.get('num_train_epochs', 3),
            max_steps=self.training_config.get('max_steps', -1),
            warmup_ratio=self.training_config.get('warmup_ratio', 0.03),
            
            # Checkpointing
            save_steps=self.training_config.get('save_steps', 500),
            save_total_limit=self.training_config.get('save_total_limit', 3),
            
            # Evaluation
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self.training_config.get('eval_steps', 500) if eval_dataset is not None else None,
            
            # Logging
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=self.training_config.get('logging_steps', 100),
            
            # Precision
            bf16=self.training_config.get('bf16', True),
            
            # Learning rate schedule
            lr_scheduler_type=self.training_config.get('lr_scheduler_type', "cosine"),
            
            # Distributed training
            local_rank=self.local_rank,
            
            # Tracking
            report_to=self.tracking_config.get('report_to', ["tensorboard"]),
        )
        
        # Set up early stopping if requested
        callbacks = []
        if self.training_config.get('early_stopping_patience', 0) > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.get('early_stopping_patience'),
                    early_stopping_threshold=self.training_config.get('early_stopping_threshold', 0.0)
                )
            )
        
        # Create the trainer
        trainer = Trainer(
            model=self.llama_wrapper.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.llama_wrapper.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        return trainer
    
    def train(self, train_dataset, eval_dataset=None):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Training results
        """
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Starting fine-tuning")
        train_result = trainer.train()
        
        # Save the model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        save_adapter_only = self.lora_config.get('use_lora', True) or self.qlora_config.get('use_qlora', False)
        trainer.save_model(self.output_dir)
        
        # Log training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Run final evaluation if evaluation dataset is provided
        if eval_dataset is not None:
            logger.info("Running final evaluation")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        
        return train_result
    
    def evaluate(self, test_dataset):
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        # Create trainer for evaluation
        trainer = self.create_trainer(None, test_dataset)
        
        # Run evaluation
        logger.info("Evaluating model")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("test", eval_metrics)
        trainer.save_metrics("test", eval_metrics)
        
        return eval_metrics
    
    def generate(self, prompts, **kwargs):
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        # Get defaults from config
        generation_config = self.evaluation_config.get('generate', {})
        
        # Override with provided kwargs
        generation_params = {**generation_config, **kwargs}
        
        return self.llama_wrapper.generate(prompts, **generation_params)
