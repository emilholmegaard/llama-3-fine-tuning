"""
Fine-tuning utilities for Llama models using LoRA and QLoRA.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    get_scheduler
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

from src.models.llama_wrapper import LlamaWrapper
from src.models.checkpoint_handler import CheckpointHandler

logger = logging.getLogger(__name__)

class GradientClippingCallback(TrainerCallback):
    """Callback for gradient clipping during training."""
    
    def __init__(self, max_grad_norm: float = 1.0):
        self.max_grad_norm = max_grad_norm
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs) -> TrainerControl:
        """Clip gradients after optimizer.step()."""
        if hasattr(kwargs, "model") and kwargs["model"].parameters():
            torch.nn.utils.clip_grad_norm_(
                kwargs["model"].parameters(), 
                self.max_grad_norm
            )
        return control

class CheckpointCallback(TrainerCallback):
    """Callback for saving checkpoints with the CheckpointHandler."""
    
    def __init__(self, checkpoint_handler: CheckpointHandler):
        self.checkpoint_handler = checkpoint_handler
    
    def on_save(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, **kwargs) -> TrainerControl:
        """Save checkpoints using CheckpointHandler."""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        
        metrics = state.log_history[-1] if state.log_history else {}
        
        # Save checkpoint
        self.checkpoint_handler.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            step=state.global_step,
            epoch=state.epoch,
            metrics=metrics
        )
        
        return control

class LlamaFineTuner:
    """
    Fine-tuning class for Llama models using PEFT methods (LoRA and QLoRA).
    Supports different optimizers, checkpoint handling, and advanced training configurations.
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
        self.setup_checkpoint_handler()
        
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
        self.checkpoint_config = self.config.get('checkpoint', {})
        self.optimizer_config = self.config.get('optimizer', {})
        
        # Set important paths
        self.base_model_name = self.model_config.get('base_model', 'meta-llama/Llama-3.3-8B')
        self.output_dir = self.model_config.get('output_dir', 'data/models/finetuned-model/')
        
        logger.info(f"Configuration loaded from {config_path}")
        
    def setup_checkpoint_handler(self) -> None:
        """Set up the checkpoint handler for saving and loading training state."""
        checkpoint_dir = self.checkpoint_config.get('dir', os.path.join(self.output_dir, 'checkpoints'))
        max_checkpoints = self.checkpoint_config.get('max_checkpoints', 3)
        save_optimizer_state = self.checkpoint_config.get('save_optimizer_state', True)
        
        self.checkpoint_handler = CheckpointHandler(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            save_optimizer_state=save_optimizer_state
        )
        
        logger.info(f"Checkpoint handler initialized at {checkpoint_dir}")
        
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
    
    def _create_optimizer(self, model) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimizer instance
        """
        optimizer_type = self.optimizer_config.get('type', 'adamw')
        lr = self.training_config.get('learning_rate', 2e-5)
        weight_decay = self.optimizer_config.get('weight_decay', 0.01)
        
        # Get trainable parameters
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(self.optimizer_config.get('beta1', 0.9), 
                       self.optimizer_config.get('beta2', 0.999)),
                eps=self.optimizer_config.get('epsilon', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    optimizer_grouped_parameters,
                    lr=lr,
                    betas=(self.optimizer_config.get('beta1', 0.9), 
                           self.optimizer_config.get('beta2', 0.99)),
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer requested but not installed. Using AdamW instead.")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    weight_decay=weight_decay
                )
        elif optimizer_type.lower() == 'adafactor':
            try:
                from transformers.optimization import Adafactor
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=lr,
                    scale_parameter=self.optimizer_config.get('scale_parameter', False),
                    relative_step=self.optimizer_config.get('relative_step', False),
                    warmup_init=self.optimizer_config.get('warmup_init', False),
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Adafactor optimizer requested but not available. Using AdamW instead.")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    weight_decay=weight_decay
                )
        else:
            logger.warning(f"Unknown optimizer type {optimizer_type}. Using AdamW instead.")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def create_trainer(self, train_dataset, eval_dataset=None, resume_from_checkpoint=None):
        """
        Create a Trainer instance for fine-tuning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path or checkpoint ID to resume from (optional)
            
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
            fp16=self.training_config.get('fp16', False),
            
            # Learning rate schedule
            lr_scheduler_type=self.training_config.get('lr_scheduler_type', "cosine"),
            
            # Distributed training
            local_rank=self.local_rank,
            
            # Gradient clipping
            max_grad_norm=self.training_config.get('max_grad_norm', 1.0),
            
            # Optimizer
            optim=self.optimizer_config.get('type', 'adamw'),
            
            # Mixed precision training
            mixed_precision=self.training_config.get('mixed_precision', None),
            
            # Reporting
            report_to=self.tracking_config.get('report_to', ["tensorboard"]),
            
            # Gradient checkpointing
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
        )
        
        # Set up callbacks
        callbacks = []
        
        # Add early stopping if requested
        if self.training_config.get('early_stopping_patience', 0) > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.get('early_stopping_patience'),
                    early_stopping_threshold=self.training_config.get('early_stopping_threshold', 0.0)
                )
            )
        
        # Add gradient clipping if requested
        if self.training_config.get('use_gradient_clipping', True):
            callbacks.append(
                GradientClippingCallback(
                    max_grad_norm=self.training_config.get('max_grad_norm', 1.0)
                )
            )
        
        # Add checkpoint callback
        callbacks.append(CheckpointCallback(self.checkpoint_handler))
        
        # Create custom optimizer if specified
        optimizer = None
        if self.optimizer_config.get('use_custom_optimizer', False):
            optimizer = self._create_optimizer(self.llama_wrapper.model)
        
        # Create the trainer
        trainer = Trainer(
            model=self.llama_wrapper.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.llama_wrapper.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, None) if optimizer else (None, None)
        )
        
        # Load checkpoint if requested
        if resume_from_checkpoint:
            logger.info(f"Attempting to resume from checkpoint: {resume_from_checkpoint}")
            
            # Check if it's a checkpoint ID
            checkpoint_found = False
            if os.path.exists(resume_from_checkpoint):
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                checkpoint_found = True
            else:
                # Try to load from checkpoint handler
                model, optimizer_state, scheduler_state, checkpoint_state = self.checkpoint_handler.load_checkpoint(
                    resume_from_checkpoint,
                    self.llama_wrapper.model
                )
                
                if checkpoint_state:
                    logger.info(f"Resuming from checkpoint {resume_from_checkpoint}")
                    # Set the model state
                    self.llama_wrapper.model = model
                    trainer.model = model
                    
                    # Set starting epoch/step in trainer
                    trainer.state.epoch = checkpoint_state.get("epoch", 0)
                    trainer.state.global_step = checkpoint_state.get("step", 0)
                    
                    checkpoint_found = True
            
            if not checkpoint_found:
                logger.warning(f"Checkpoint {resume_from_checkpoint} not found, starting training from scratch")
        
        return trainer
    
    def train(self, train_dataset, eval_dataset=None, resume_from_checkpoint=None):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path or checkpoint ID to resume from (optional)
            
        Returns:
            Training results
        """
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset, resume_from_checkpoint)
        
        # Start training
        logger.info("Starting fine-tuning")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint if os.path.exists(resume_from_checkpoint) else None)
        
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
        
    def save_adapter(self, output_path=None):
        """
        Save the LoRA adapter separately.
        
        Args:
            output_path: Path to save the adapter (optional)
            
        Returns:
            Path to the saved adapter
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "adapter")
        
        os.makedirs(output_path, exist_ok=True)
        
        if hasattr(self.llama_wrapper.model, "save_pretrained"):
            logger.info(f"Saving LoRA adapter to {output_path}")
            self.llama_wrapper.model.save_pretrained(output_path)
            
            # Save the tokenizer for convenience
            self.llama_wrapper.tokenizer.save_pretrained(output_path)
            
            return output_path
        else:
            logger.error("Model does not support save_pretrained")
            return None
    
    def load_adapter(self, adapter_path):
        """
        Load a LoRA adapter.
        
        Args:
            adapter_path: Path to the adapter
            
        Returns:
            The model with loaded adapter
        """
        if not os.path.exists(adapter_path):
            logger.error(f"Adapter path {adapter_path} does not exist")
            return None
        
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        
        if isinstance(self.llama_wrapper.model, PeftModel):
            # Already a PEFT model, load adapter
            self.llama_wrapper.model.load_adapter(adapter_path)
        else:
            # Convert to PEFT model and load adapter
            self.llama_wrapper.model = PeftModel.from_pretrained(
                self.llama_wrapper.model,
                adapter_path,
                is_trainable=True
            )
        
        return self.llama_wrapper.model
