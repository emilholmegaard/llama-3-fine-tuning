"""
Fine-tuning utilities for Llama models using LoRA and QLoRA.
"""

import os
import logging
import yaml
import json
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
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        """Apply gradient clipping after backward pass but before optimizer step."""
        if "model" in kwargs and "optimizer" in kwargs:
            # Get model and optimizer from kwargs
            model = kwargs["model"]
            optimizer = kwargs["optimizer"]
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=self.max_grad_norm
            )
            
            logger.debug(f"Gradient clipped with max norm: {self.max_grad_norm}")
        
        return control

class CustomCheckpointCallback(TrainerCallback):
    """Callback for custom checkpoint handling."""
    
    def __init__(self, checkpoint_handler: CheckpointHandler, checkpoint_steps: int = 500):
        self.checkpoint_handler = checkpoint_handler
        self.checkpoint_steps = checkpoint_steps
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        """Save checkpoint at specified intervals."""
        if state.global_step % self.checkpoint_steps == 0:
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            
            # Get LR scheduler if available
            scheduler = None
            if "lr_scheduler" in kwargs:
                scheduler = kwargs["lr_scheduler"]
            
            # Get metrics
            metrics = {
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
                "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None
            }
            
            # Save checkpoint
            if model is not None:
                self.checkpoint_handler.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=state.global_step,
                    epoch=state.epoch,
                    metrics=metrics,
                    extra_data={"args": args.to_dict()}
                )
                
                logger.info(f"Saved checkpoint at step {state.global_step}")
        
        return control

class LlamaFineTuner:
    """
    Fine-tuning class for Llama models using PEFT methods (LoRA and QLoRA).
    """
    
    def __init__(
        self,
        config_path: str,
        local_rank: int = -1,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Initialize the fine-tuner from a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            local_rank: Local rank for distributed training
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
        """
        self.local_rank = local_rank
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_config(config_path)
        
        # Initialize checkpoint handler
        checkpoint_dir = self.training_config.get("checkpoint_dir") or os.path.join(self.output_dir, "checkpoints")
        max_checkpoints = self.training_config.get("checkpoint_keep_limit", 3)
        self.checkpoint_handler = CheckpointHandler(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            save_optimizer_state=True
        )
        
        # Setup model
        self.setup_model()
        
        # Initialize optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.scheduler = None
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = self._resolve_checkpoint_path(self.resume_from_checkpoint)
            if self.resume_from_checkpoint:
                self.resume()
        
    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Optional[str]:
        """
        Resolve checkpoint path, handling special values like 'latest'.
        
        Args:
            checkpoint_path: Path or special value
            
        Returns:
            Resolved path or None if not found
        """
        if checkpoint_path == "latest":
            # Use metadata to find the latest checkpoint
            latest_checkpoint_id = self.checkpoint_handler.metadata.get("last_checkpoint")
            if latest_checkpoint_id:
                for checkpoint in self.checkpoint_handler.metadata.get("checkpoints", []):
                    if checkpoint.get("id") == latest_checkpoint_id:
                        return checkpoint.get("path")
            return None
        
        # If it's a path, return as is
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        # If it's a checkpoint ID, resolve to path
        checkpoint_id_path = self.checkpoint_handler._get_checkpoint_path(checkpoint_path)
        if os.path.exists(checkpoint_id_path):
            return checkpoint_id_path
        
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
        
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
        self.scheduler_config = self.config.get('scheduler', {})
        
        # Load hyperparameter optimization config if available
        self.hyperopt_config = self.config.get('hyperopt', {})
        
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
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        # Get optimizer parameters
        optimizer_name = self.optimizer_config.get('name', 'adamw').lower()
        
        # Get trainable parameters
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.llama_wrapper.model.named_parameters() 
                       if p.requires_grad and not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.llama_wrapper.model.named_parameters() 
                         if p.requires_grad and any(nd in n for nd in no_decay)]
        
        # Get weight decay
        weight_decay = self.optimizer_config.get('weight_decay', 0.01)
        
        # Get learning rate
        learning_rate = self.training_config.get('learning_rate', 2e-5)
        
        # Define optimizer groups
        optim_groups = [
            {"params": params_decay, "weight_decay": weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0}
        ]
        
        # Create optimizer based on name
        if optimizer_name == 'adamw':
            from transformers.optimization import AdamW
            optimizer = AdamW(
                optim_groups,
                lr=learning_rate,
                betas=(self.optimizer_config.get('adam_beta1', 0.9), 
                       self.optimizer_config.get('adam_beta2', 0.999)),
                eps=self.optimizer_config.get('adam_epsilon', 1e-8),
            )
        elif optimizer_name == 'adafactor':
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                optim_groups,
                lr=learning_rate,
                scale_parameter=self.optimizer_config.get('scale_parameter', False),
                relative_step=self.optimizer_config.get('relative_step', False),
                warmup_init=self.optimizer_config.get('warmup_init', False),
            )
        elif optimizer_name == 'lion':
            # If using Lion, make sure it's installed
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    optim_groups,
                    lr=learning_rate,
                    betas=(self.optimizer_config.get('lion_beta1', 0.9), 
                           self.optimizer_config.get('lion_beta2', 0.99)),
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not found, falling back to AdamW. "
                             "Install with: pip install lion-pytorch")
                from transformers.optimization import AdamW
                optimizer = AdamW(optim_groups, lr=learning_rate)
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, falling back to AdamW")
            from transformers.optimization import AdamW
            optimizer = AdamW(optim_groups, lr=learning_rate)
        
        logger.info(f"Created {optimizer_name} optimizer with learning rate {learning_rate}")
        return optimizer
    
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
            fp16=self.training_config.get('fp16', False),
            
            # Learning rate schedule
            lr_scheduler_type=self.scheduler_config.get('type', "cosine"),
            
            # Distributed training
            local_rank=self.local_rank,
            
            # Tracking
            report_to=self.tracking_config.get('report_to', ["tensorboard"]),
            
            # Misc
            seed=self.training_config.get('seed', 42),
            data_seed=self.training_config.get('data_seed', 42),
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
        if self.training_config.get('gradient_clipping', False):
            max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
            callbacks.append(GradientClippingCallback(max_grad_norm=max_grad_norm))
            logger.info(f"Enabled gradient clipping with max norm {max_grad_norm}")
        
        # Add checkpoint callback if requested
        if self.training_config.get('use_custom_checkpointing', True):
            checkpoint_steps = self.training_config.get('checkpoint_save_steps', 500)
            callbacks.append(
                CustomCheckpointCallback(
                    checkpoint_handler=self.checkpoint_handler,
                    checkpoint_steps=checkpoint_steps
                )
            )
            logger.info(f"Enabled custom checkpointing every {checkpoint_steps} steps")
        
        # Create optimizer if configured
        optimizer = self.optimizer if self.optimizer else self._create_optimizer()
        
        # Create the trainer
        trainer = Trainer(
            model=self.llama_wrapper.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.llama_wrapper.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, self.scheduler)  # Pass optimizer and scheduler
        )
        
        return trainer
    
    def resume(self):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
        
        # Check if checkpoint path exists
        if not os.path.exists(self.resume_from_checkpoint):
            logger.warning(f"Checkpoint path {self.resume_from_checkpoint} not found")
            return False
        
        # Create optimizer
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Load model and training state from checkpoint
        model, optimizer, scheduler, checkpoint_state = self.checkpoint_handler.load_checkpoint(
            checkpoint_id=os.path.basename(self.resume_from_checkpoint),
            model=self.llama_wrapper.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        
        # Update model and optimizer
        self.llama_wrapper.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Log resumption info
        if checkpoint_state:
            logger.info(f"Resumed from step {checkpoint_state.get('step')} "
                      f"epoch {checkpoint_state.get('epoch')}")
            return True
        else:
            logger.warning("Checkpoint state not found or empty")
            return False
    
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
        train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # Save the model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        save_adapter_only = self.lora_config.get('use_lora', True) or self.qlora_config.get('use_qlora', False)
        trainer.save_model(self.output_dir)
        
        # Save training configuration
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
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
