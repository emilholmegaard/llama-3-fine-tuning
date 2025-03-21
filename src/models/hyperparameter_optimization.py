"""
Hyperparameter optimization utilities for LoRA fine-tuning.
Supports Optuna and Ray Tune for hyperparameter search.
"""

import os
import logging
import tempfile
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from datasets import Dataset, DatasetDict
import torch

from src.models.fine_tuning import LlamaFineTuner

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization for LoRA fine-tuning using Optuna or Ray Tune.
    """
    
    def __init__(
        self, 
        base_config_path: str,
        backend: str = "optuna",
        n_trials: int = 10,
        timeout: Optional[int] = None,
        study_name: str = "llama-lora-optimization",
        direction: str = "minimize",
        metric: str = "eval_loss",
        seed: int = 42,
        ray_address: Optional[str] = None,
        n_gpus_per_trial: float = 1.0
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            base_config_path: Path to base YAML configuration file
            backend: Optimization backend ('optuna' or 'ray')
            n_trials: Number of trials to run
            timeout: Maximum time in seconds for the study (optional)
            study_name: Name of the optimization study
            direction: Direction of optimization ('minimize' or 'maximize')
            metric: Metric to optimize
            seed: Random seed for reproducibility
            ray_address: Address of Ray cluster (optional, for distributed optimization)
            n_gpus_per_trial: Number of GPUs per trial (can be fractional)
        """
        self.base_config_path = base_config_path
        self.backend = backend
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.direction = direction
        self.metric = metric
        self.seed = seed
        self.ray_address = ray_address
        self.n_gpus_per_trial = n_gpus_per_trial
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Set default search space
        self.search_space = self._default_search_space()
        
        logger.info(f"Hyperparameter optimizer initialized with {backend} backend")
        
    def _default_search_space(self) -> Dict[str, Any]:
        """
        Define default search space for hyperparameters.
        
        Returns:
            Dictionary with search space definitions
        """
        return {
            "learning_rate": {
                "type": "loguniform",
                "low": 1e-6,
                "high": 1e-4
            },
            "lora_r": {
                "type": "int",
                "low": 4,
                "high": 32
            },
            "lora_alpha": {
                "type": "int",
                "low": 8,
                "high": 64
            },
            "lora_dropout": {
                "type": "uniform",
                "low": 0.0,
                "high": 0.3
            },
            "batch_size": {
                "type": "categorical",
                "choices": [1, 2, 4, 8]
            },
            "gradient_accumulation_steps": {
                "type": "categorical",
                "choices": [1, 2, 4, 8, 16]
            },
            "warmup_ratio": {
                "type": "uniform",
                "low": 0.01,
                "high": 0.1
            }
        }
        
    def set_search_space(self, search_space: Dict[str, Any]) -> None:
        """
        Set custom search space for hyperparameters.
        
        Args:
            search_space: Dictionary with search space definitions
        """
        self.search_space = search_space
        logger.info("Custom search space set for hyperparameter optimization")
    
    def _create_trial_config(self, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create configuration for a specific trial.
        
        Args:
            trial_params: Parameters for this trial
            
        Returns:
            Complete configuration dictionary for this trial
        """
        # Deep copy the base config to avoid modifying it
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Update learning rate
        if "learning_rate" in trial_params:
            config["training"]["learning_rate"] = trial_params["learning_rate"]
            
        # Update batch size and gradient accumulation steps
        if "batch_size" in trial_params:
            config["training"]["batch_size"] = trial_params["batch_size"]
        if "gradient_accumulation_steps" in trial_params:
            config["training"]["gradient_accumulation_steps"] = trial_params["gradient_accumulation_steps"]
            
        # Update warmup ratio
        if "warmup_ratio" in trial_params:
            config["training"]["warmup_ratio"] = trial_params["warmup_ratio"]
            
        # Update LoRA parameters
        if "lora_r" in trial_params:
            config["lora"]["r"] = trial_params["lora_r"]
        if "lora_alpha" in trial_params:
            config["lora"]["alpha"] = trial_params["lora_alpha"]
        if "lora_dropout" in trial_params:
            config["lora"]["dropout"] = trial_params["lora_dropout"]
                
        return config
    
    def _objective(self, 
                  trial: optuna.Trial, 
                  train_dataset: Dataset, 
                  eval_dataset: Optional[Dataset] = None) -> float:
        """
        Objective function for Optuna trials.
        
        Args:
            trial: Optuna trial object
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Metric value to optimize
        """
        # Sample parameters for this trial
        params = {}
        
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config["low"], 
                    param_config["high"], 
                    log=True
                )
            elif param_config["type"] == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config["choices"]
                )
        
        # Create config for this trial
        trial_config = self._create_trial_config(params)
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
            yaml.dump(trial_config, temp_config)
            temp_config_path = temp_config.name
        
        try:
            # Initialize fine-tuner with trial config
            # Set a unique output directory for this trial
            trial_output_dir = os.path.join(
                self.base_config["model"]["output_dir"],
                f"trial_{trial.number}"
            )
            trial_config["model"]["output_dir"] = trial_output_dir
            
            # Create fine-tuner
            fine_tuner = LlamaFineTuner(temp_config_path)
            
            # Train model
            train_result = fine_tuner.train(train_dataset, eval_dataset)
            
            # Get the metric we're optimizing
            metric_value = None
            if self.metric.startswith("eval_") and eval_dataset is not None:
                # Get evaluation metrics
                metric_value = train_result.metrics.get(self.metric)
            else:
                # Get training metrics
                metric_value = train_result.metrics.get(self.metric)
            
            if metric_value is None:
                logger.warning(f"Metric {self.metric} not found, using fallback value")
                # Fallback to eval_loss or train_loss
                metric_value = train_result.metrics.get(
                    "eval_loss" if eval_dataset is not None else "train_loss"
                )
                
            return metric_value
        
        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def _ray_objective(self, 
                      config: Dict[str, Any],
                      train_dataset: Dataset,
                      eval_dataset: Optional[Dataset] = None) -> None:
        """
        Objective function for Ray Tune trials.
        
        Args:
            config: Configuration for this trial
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        # Create config for this trial
        trial_config = self._create_trial_config(config)
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
            yaml.dump(trial_config, temp_config)
            temp_config_path = temp_config.name
        
        try:
            # Initialize fine-tuner with trial config
            fine_tuner = LlamaFineTuner(temp_config_path)
            
            # Train model
            train_result = fine_tuner.train(train_dataset, eval_dataset)
            
            # Get the metric we're optimizing
            metric_value = None
            if self.metric.startswith("eval_") and eval_dataset is not None:
                # Get evaluation metrics
                metric_value = train_result.metrics.get(self.metric)
            else:
                # Get training metrics
                metric_value = train_result.metrics.get(self.metric)
            
            if metric_value is None:
                logger.warning(f"Metric {self.metric} not found, using fallback value")
                # Fallback to eval_loss or train_loss
                metric_value = train_result.metrics.get(
                    "eval_loss" if eval_dataset is not None else "train_loss"
                )
            
            # Report metrics to Ray Tune
            tune.report({self.metric: metric_value})
            
        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def optimize(self, 
                train_dataset: Dataset, 
                eval_dataset: Optional[Dataset] = None,
                optimization_output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            optimization_output_dir: Directory to save optimization results (optional)
            
        Returns:
            Best parameters found
        """
        if optimization_output_dir is None:
            optimization_output_dir = os.path.join(
                self.base_config["model"]["output_dir"],
                "hpo_results"
            )
        
        os.makedirs(optimization_output_dir, exist_ok=True)
        
        if self.backend == "optuna":
            return self._run_optuna_optimization(
                train_dataset, 
                eval_dataset,
                optimization_output_dir
            )
        elif self.backend == "ray":
            return self._run_ray_optimization(
                train_dataset, 
                eval_dataset,
                optimization_output_dir
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _run_optuna_optimization(self, 
                               train_dataset: Dataset, 
                               eval_dataset: Optional[Dataset] = None,
                               optimization_output_dir: str = None) -> Dict[str, Any]:
        """
        Run optimization using Optuna.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            optimization_output_dir: Directory to save optimization results
            
        Returns:
            Best parameters found
        """
        logger.info("Starting hyperparameter optimization with Optuna")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, train_dataset, eval_dataset),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        # Save study results
        if optimization_output_dir:
            os.makedirs(optimization_output_dir, exist_ok=True)
            
            # Save best parameters
            best_params_path = os.path.join(optimization_output_dir, "best_params.json")
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Save study statistics
            study_path = os.path.join(optimization_output_dir, "study.pkl")
            try:
                import joblib
                joblib.dump(study, study_path)
                logger.info(f"Study saved to {study_path}")
            except Exception as e:
                logger.warning(f"Failed to save study: {e}")
                
            # Generate and save best configuration
            best_config = self._create_trial_config(best_params)
            best_config_path = os.path.join(optimization_output_dir, "best_config.yaml")
            with open(best_config_path, 'w') as f:
                yaml.dump(best_config, f)
            logger.info(f"Best configuration saved to {best_config_path}")
        
        return best_params
    
    def _run_ray_optimization(self, 
                            train_dataset: Dataset, 
                            eval_dataset: Optional[Dataset] = None,
                            optimization_output_dir: str = None) -> Dict[str, Any]:
        """
        Run optimization using Ray Tune.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            optimization_output_dir: Directory to save optimization results
            
        Returns:
            Best parameters found
        """
        logger.info("Starting hyperparameter optimization with Ray Tune")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init()
        
        # Convert search space to Ray format
        ray_search_space = {}
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "loguniform":
                ray_search_space[param_name] = tune.loguniform(
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_config["type"] == "uniform":
                ray_search_space[param_name] = tune.uniform(
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_config["type"] == "int":
                ray_search_space[param_name] = tune.randint(
                    param_config["low"], 
                    param_config["high"] + 1
                )
            elif param_config["type"] == "categorical":
                ray_search_space[param_name] = tune.choice(
                    param_config["choices"]
                )
        
        # Setup Ray Tune optimizer
        optuna_search = OptunaSearch(
            metric=self.metric,
            mode=self.direction,
            seed=self.seed
        )
        
        # Setup ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            metric=self.metric,
            mode=self.direction,
            max_t=self.base_config["training"].get("num_train_epochs", 3),
            grace_period=1,
            reduction_factor=2
        )
        
        # Define callback function to pass datasets to the objective
        def training_function(config):
            self._ray_objective(config, train_dataset, eval_dataset)
        
        # Run Ray Tune
        analysis = tune.run(
            training_function,
            config=ray_search_space,
            metric=self.metric,
            mode=self.direction,
            resources_per_trial={"gpu": self.n_gpus_per_trial},
            num_samples=self.n_trials,
            search_alg=optuna_search,
            scheduler=scheduler,
            local_dir=optimization_output_dir,
            name=self.study_name,
            fail_fast=True
        )
        
        # Get best parameters
        best_params = analysis.best_config
        logger.info(f"Best parameters: {best_params}")
        
        # Save best configuration
        if optimization_output_dir:
            # Generate and save best configuration
            best_config = self._create_trial_config(best_params)
            best_config_path = os.path.join(optimization_output_dir, "best_config.yaml")
            with open(best_config_path, 'w') as f:
                yaml.dump(best_config, f)
            logger.info(f"Best configuration saved to {best_config_path}")
        
        return best_params
