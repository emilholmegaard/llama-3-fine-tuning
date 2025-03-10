"""
Evaluator for fine-tuned language models.

This module provides a class for evaluating fine-tuned models,
running inference on test datasets, and calculating various metrics.
"""

import os
import json
import logging
import time
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextGenerationPipeline
)

from src.evaluation.metrics import (
    calculate_perplexity,
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_exact_match,
    calculate_f1_score,
    calculate_accuracy,
    calculate_domain_specific_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationExample:
    """Class to store evaluation examples and results."""
    input_text: str
    reference_text: str
    generated_text: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class ModelEvaluator:
    """
    Evaluator for fine-tuned language models.
    
    This class provides methods for loading models, running inference,
    calculating various evaluation metrics, and visualizing results.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        tensor_parallel: bool = False,
        dtype: Optional[torch.dtype] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model or model identifier from HuggingFace
            device: Device to run the model on ('auto', 'cuda', 'cpu')
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            tensor_parallel: Whether to use tensor parallelism for multi-GPU
            dtype: Data type for model weights (e.g., torch.float16)
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join("data", "evaluation", os.path.basename(model_path))
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with specified precision
        model_kwargs = {}
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        if tensor_parallel and torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "auto"
            
        if dtype:
            model_kwargs["torch_dtype"] = dtype
        elif self.device == "cuda":
            # Default to half precision on GPU
            model_kwargs["torch_dtype"] = torch.float16
            
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if self.device == "cuda" and "device_map" not in model_kwargs:
            self.model.to(self.device)
            
        # Create generation pipeline
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Initialize results storage
        self.evaluation_results = {}
        self.examples = []
        
    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        data_files: Optional[Union[str, List[str], Dict[str, str]]] = None,
        split: str = "test",
        input_key: str = "prompt",
        reference_key: str = "completion",
        **kwargs
    ) -> Dataset:
        """
        Load evaluation dataset.
        
        Args:
            dataset_path: Path to local dataset directory
            dataset_name: Name of dataset on Hugging Face Hub
            data_files: Path(s) to data files
            split: Dataset split to use
            input_key: Key for input texts in the dataset
            reference_key: Key for reference texts in the dataset
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            The loaded dataset
        """
        if dataset_path or dataset_name:
            logger.info(f"Loading dataset from {'hub' if dataset_name else 'local path'}")
            dataset = load_dataset(
                path=dataset_path or dataset_name,
                data_files=data_files,
                split=split,
                **kwargs
            )
        elif data_files:
            logger.info(f"Loading dataset from files: {data_files}")
            dataset = load_dataset(
                "json" if isinstance(data_files, (str, list)) and any(f.endswith(".json") or f.endswith(".jsonl") for f in ([data_files] if isinstance(data_files, str) else data_files)) else "csv",
                data_files=data_files,
                split="train",  # Will be the only split when loading from files
                **kwargs
            )
        else:
            raise ValueError("Either dataset_path, dataset_name, or data_files must be provided")
            
        # Validate keys
        if input_key not in dataset.column_names:
            raise ValueError(f"Input key '{input_key}' not found in dataset columns: {dataset.column_names}")
        if reference_key not in dataset.column_names:
            raise ValueError(f"Reference key '{reference_key}' not found in dataset columns: {dataset.column_names}")
            
        self.dataset = dataset
        self.input_key = input_key
        self.reference_key = reference_key
        
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        return dataset
    
    def evaluate_perplexity(
        self,
        dataset: Optional[Dataset] = None,
        text_key: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 1024
    ) -> Dict[str, float]:
        """
        Evaluate model perplexity on a dataset.
        
        Args:
            dataset: Dataset to evaluate on (defaults to self.dataset)
            text_key: Key for texts to evaluate (defaults to self.reference_key)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity results
        """
        dataset = dataset or self.dataset
        text_key = text_key or self.reference_key
        
        if not dataset:
            raise ValueError("No dataset provided for evaluation")
            
        logger.info("Calculating perplexity...")
        texts = dataset[text_key]
        
        results = calculate_perplexity(
            self.model,
            self.tokenizer,
            texts,
            batch_size=batch_size,
            device=self.device,
            max_length=max_length
        )
        
        logger.info(f"Perplexity: {results['perplexity']:.4f}")
        
        # Store results
        self.evaluation_results["perplexity"] = results
        
        return results
    
    def generate_text(
        self,
        dataset: Optional[Dataset] = None,
        input_key: Optional[str] = None,
        reference_key: Optional[str] = None,
        num_examples: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        save_examples: bool = True
    ) -> List[EvaluationExample]:
        """
        Generate text for evaluation examples.
        
        Args:
            dataset: Dataset to generate from (defaults to self.dataset)
            input_key: Key for input texts (defaults to self.input_key)
            reference_key: Key for reference texts (defaults to self.reference_key)
            num_examples: Number of examples to generate (None for all)
            generation_config: Configuration for text generation
            save_examples: Whether to save generated examples
            
        Returns:
            List of EvaluationExample objects
        """
        dataset = dataset or self.dataset
        input_key = input_key or self.input_key
        reference_key = reference_key or self.reference_key
        
        if not dataset:
            raise ValueError("No dataset provided for text generation")
            
        # Select subset of examples if requested
        if num_examples and num_examples < len(dataset):
            indices = np.random.choice(len(dataset), num_examples, replace=False)
            dataset = dataset.select(indices)
            
        # Default generation config
        default_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Update with user config
        gen_config = {**default_config, **(generation_config or {})}
        
        # Create generation config
        generation_config = GenerationConfig(**gen_config)
        
        examples = []
        logger.info(f"Generating text for {len(dataset)} examples...")
        
        start_time = time.time()
        for i, example in enumerate(dataset):
            input_text = example[input_key]
            reference_text = example[reference_key]
            
            # Generate text
            outputs = self.pipeline(
                input_text,
                generation_config=generation_config,
                return_full_text=False  # Don't include the prompt in the output
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Create evaluation example
            eval_example = EvaluationExample(
                input_text=input_text,
                reference_text=reference_text,
                generated_text=generated_text
            )
            
            examples.append(eval_example)
            
            # Log progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                examples_per_sec = (i + 1) / elapsed
                logger.info(f"Generated {i + 1}/{len(dataset)} examples ({examples_per_sec:.2f} examples/sec)")
                
        # Save examples if requested
        if save_examples:
            self.examples = examples
            self._save_examples(examples)
            
        logger.info(f"Generated {len(examples)} examples in {time.time() - start_time:.2f} seconds")
        return examples
    
    def calculate_metrics(
        self,
        examples: Optional[List[EvaluationExample]] = None,
        metrics: List[str] = ["rouge", "bleu", "exact_match", "f1"],
        domain_type: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for generated examples.
        
        Args:
            examples: List of evaluation examples (defaults to self.examples)
            metrics: List of metrics to calculate
            domain_type: Domain-specific metrics to include
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with evaluation metrics
        """
        examples = examples or self.examples
        
        if not examples:
            raise ValueError("No examples provided for metric calculation")
            
        # Extract predictions and references
        predictions = [ex.generated_text for ex in examples]
        references = [ex.reference_text for ex in examples]
        
        # Initialize results dictionary
        results = {}
        
        # Calculate requested metrics
        for metric in metrics:
            logger.info(f"Calculating {metric} scores...")
            
            if metric == "rouge":
                rouge_scores = calculate_rouge_scores(predictions, references)
                results["rouge"] = rouge_scores
                
            elif metric == "bleu":
                # Format references for BLEU
                references_for_bleu = [[ref] for ref in references]
                bleu_scores = calculate_bleu_score(predictions, references_for_bleu)
                results["bleu"] = bleu_scores
                
            elif metric == "exact_match":
                em_score = calculate_exact_match(predictions, references)
                results["exact_match"] = em_score
                
            elif metric == "f1":
                f1_scores = calculate_f1_score(predictions, references, normalize_text_inputs=True)
                results["f1"] = f1_scores
            
            # Add more metrics as needed
                
        # Calculate domain-specific metrics if provided
        if domain_type:
            logger.info(f"Calculating domain-specific metrics for '{domain_type}'...")
            domain_metrics = calculate_domain_specific_metrics(
                predictions, references, domain_type=domain_type
            )
            results["domain"] = domain_metrics
            
        # Update evaluation results
        self.evaluation_results.update(results)
        
        # Calculate metrics for individual examples
        for i, example in enumerate(examples):
            example_metrics = {}
            
            # Add per-example metrics here as needed
            if "exact_match" in results:
                example_metrics["exact_match"] = 1.0 if predictions[i] == references[i] else 0.0
                
            if "domain" in results and domain_type == "qa":
                # For QA tasks, add token-level F1 per example
                pred_tokens = set(predictions[i].lower().split())
                ref_tokens = set(references[i].lower().split())
                
                if not pred_tokens and not ref_tokens:
                    example_metrics["token_f1"] = 1.0
                elif not pred_tokens or not ref_tokens:
                    example_metrics["token_f1"] = 0.0
                else:
                    common_tokens = pred_tokens.intersection(ref_tokens)
                    precision = len(common_tokens) / len(pred_tokens)
                    recall = len(common_tokens) / len(ref_tokens)
                    example_metrics["token_f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            
            # Update example metrics
            examples[i].metrics = example_metrics
            
        # Save results if requested
        if save_results:
            self._save_results(results)
            self._save_examples(examples)  # Update with metrics
            
        logger.info("Metric calculation complete")
        return results
    
    def visualize_results(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        output_format: str = "png",
        show_plots: bool = False
    ):
        """
        Visualize evaluation results with plots.
        
        Args:
            metrics_to_plot: List of metrics to visualize
            output_format: Format for saving plots ('png', 'pdf', 'svg')
            show_plots: Whether to display plots
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results to visualize")
            return
            
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Determine which metrics to plot
        if not metrics_to_plot:
            metrics_to_plot = list(self.evaluation_results.keys())
            
        # Generate plots for each metric type
        for metric_name in metrics_to_plot:
            if metric_name not in self.evaluation_results:
                logger.warning(f"Metric '{metric_name}' not found in evaluation results")
                continue
                
            metric_data = self.evaluation_results[metric_name]
            
            if metric_name == "perplexity":
                # Perplexity is a single value
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(["Perplexity"], [metric_data["perplexity"]], color="skyblue")
                ax.set_ylabel("Perplexity")
                ax.set_title("Model Perplexity")
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"perplexity.{output_format}"))
                
            elif metric_name == "rouge":
                # Rouge scores for different types
                fig, ax = plt.subplots(figsize=(10, 6))
                rouge_types = list(metric_data.keys())
                x = np.arange(len(rouge_types))
                width = 0.25
                
                # Plot precision, recall, and F1 for each ROUGE type
                precision_vals = [metric_data[t]["precision"] for t in rouge_types]
                recall_vals = [metric_data[t]["recall"] for t in rouge_types]
                f1_vals = [metric_data[t]["fmeasure"] for t in rouge_types]
                
                ax.bar(x - width, precision_vals, width, label="Precision")
                ax.bar(x, recall_vals, width, label="Recall")
                ax.bar(x + width, f1_vals, width, label="F1")
                
                ax.set_xlabel("ROUGE Type")
                ax.set_ylabel("Score")
                ax.set_title("ROUGE Scores")
                ax.set_xticks(x)
                ax.set_xticklabels(rouge_types)
                ax.legend()
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"rouge_scores.{output_format}"))
                
            elif metric_name == "bleu":
                # BLEU score and n-gram precisions
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot overall BLEU score
                ax.bar(["BLEU"], [metric_data["bleu"]], color="skyblue")
                
                # Plot n-gram precisions if available
                if "precisions" in metric_data:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    n_values = [f"{i+1}-gram" for i in range(len(metric_data["precisions"]))]
                    ax2.bar(n_values, metric_data["precisions"], color="lightgreen")
                    ax2.set_xlabel("N-gram")
                    ax2.set_ylabel("Precision")
                    ax2.set_title("BLEU N-gram Precisions")
                    
                    # Save n-gram plot
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"bleu_ngram_precision.{output_format}"))
                
                ax.set_ylabel("Score")
                ax.set_title("BLEU Score")
                
                # Save BLEU plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"bleu_score.{output_format}"))
                
            elif metric_name == "f1" or (metric_name == "domain" and "f1" in metric_data):
                # F1, precision, recall
                f1_data = metric_data if metric_name == "f1" else metric_data.get("f1", metric_data)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                metrics = ["precision", "recall", "f1"]
                values = [f1_data.get(m, 0) for m in metrics]
                
                ax.bar(metrics, values, color=["lightblue", "lightgreen", "coral"])
                ax.set_ylabel("Score")
                ax.set_title("Precision, Recall, and F1 Score")
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"f1_metrics.{output_format}"))
                
            # Add more metric visualizations as needed
            
            # Close plot if not showing
            if not show_plots:
                plt.close()
                
        # Create a summary plot with key metrics
        self._create_summary_plot(output_format, plots_dir, show_plots)
        
        logger.info(f"Visualization complete. Plots saved to {plots_dir}")
    
    def _create_summary_plot(self, output_format, plots_dir, show_plots):
        """Create a summary plot with the most important metrics."""
        # Extract key metrics for summary
        summary_metrics = {}
        
        if "exact_match" in self.evaluation_results:
            summary_metrics["Exact Match"] = self.evaluation_results["exact_match"]
            
        if "f1" in self.evaluation_results:
            summary_metrics["F1"] = self.evaluation_results["f1"]["f1"]
            
        if "bleu" in self.evaluation_results:
            summary_metrics["BLEU"] = self.evaluation_results["bleu"]["bleu"]
            
        if "rouge" in self.evaluation_results and "rougeL" in self.evaluation_results["rouge"]:
            summary_metrics["ROUGE-L"] = self.evaluation_results["rouge"]["rougeL"]["fmeasure"]
            
        if summary_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = list(summary_metrics.keys())
            values = list(summary_metrics.values())
            
            # Create bar chart
            bars = ax.bar(metrics, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(metrics))))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Summary of Evaluation Metrics")
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"metrics_summary.{output_format}"))
            
            if not show_plots:
                plt.close()
    
    def compare_models(
        self,
        other_evaluators: List["ModelEvaluator"],
        metrics_to_compare: Optional[List[str]] = None,
        output_format: str = "png",
        show_plots: bool = False
    ):
        """
        Compare this model with other evaluated models.
        
        Args:
            other_evaluators: List of other ModelEvaluator instances
            metrics_to_compare: List of metrics to compare
            output_format: Format for saving plots
            show_plots: Whether to display plots
        """
        # Ensure all evaluators have results
        if not self.evaluation_results:
            logger.warning("This model has no evaluation results")
            return
            
        valid_evaluators = [e for e in other_evaluators if e.evaluation_results]
        if not valid_evaluators:
            logger.warning("No other models with evaluation results for comparison")
            return
            
        # All evaluators including this one
        all_evaluators = [self] + valid_evaluators
        model_names = [os.path.basename(e.model_path) for e in all_evaluators]
        
        # Create comparison plots directory
        plots_dir = os.path.join(self.output_dir, "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Determine which metrics to compare
        if not metrics_to_compare:
            # Use common metrics across all evaluators
            metrics_sets = [set(e.evaluation_results.keys()) for e in all_evaluators]
            common_metrics = set.intersection(*metrics_sets)
            metrics_to_compare = list(common_metrics)
            
        # Generate comparison plots for each metric
        for metric_name in metrics_to_compare:
            # Skip metrics that aren't present in all evaluators
            if not all(metric_name in e.evaluation_results for e in all_evaluators):
                logger.warning(f"Metric '{metric_name}' not present in all models, skipping comparison")
                continue
                
            if metric_name == "perplexity":
                # Compare perplexity
                fig, ax = plt.subplots(figsize=(10, 6))
                perplexities = [e.evaluation_results["perplexity"]["perplexity"] for e in all_evaluators]
                
                # Create bar chart
                ax.bar(model_names, perplexities, color=plt.cm.viridis(np.linspace(0, 0.8, len(model_names))))
                ax.set_ylabel("Perplexity")
                ax.set_title("Perplexity Comparison")
                plt.xticks(rotation=45, ha="right")
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"perplexity_comparison.{output_format}"))
                
            elif metric_name == "f1":
                # Compare F1 scores
                fig, ax = plt.subplots(figsize=(10, 6))
                f1_scores = [e.evaluation_results["f1"]["f1"] for e in all_evaluators]
                
                # Create bar chart
                ax.bar(model_names, f1_scores, color=plt.cm.viridis(np.linspace(0, 0.8, len(model_names))))
                ax.set_ylabel("F1 Score")
                ax.set_title("F1 Score Comparison")
                plt.xticks(rotation=45, ha="right")
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"f1_comparison.{output_format}"))
                
            # Add more comparison plots as needed
                
            # Close plot if not showing
            if not show_plots:
                plt.close()
                
        logger.info(f"Model comparison complete. Plots saved to {plots_dir}")
    
    def _save_examples(self, examples: List[EvaluationExample]):
        """Save evaluation examples to disk."""
        examples_file = os.path.join(self.output_dir, "generated_examples.jsonl")
        
        # Convert examples to dictionaries
        example_dicts = [asdict(ex) for ex in examples]
        
        # Save as JSONL
        with open(examples_file, "w") as f:
            for example in example_dicts:
                f.write(json.dumps(example) + "\n")
                
        logger.info(f"Saved {len(examples)} examples to {examples_file}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to disk."""
        results_file = os.path.join(self.output_dir, "evaluation_results.json")
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        # Save as JSON
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Also save as CSV for easy import into other tools
        self._save_results_as_csv(results)
    
    def _save_results_as_csv(self, results: Dict[str, Any]):
        """Save evaluation results in CSV format for easy import."""
        csv_dir = os.path.join(self.output_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        # Try to flatten and save each metric type
        for metric_name, metric_data in results.items():
            if isinstance(metric_data, dict):
                if metric_name == "perplexity":
                    # Perplexity is a special case
                    df = pd.DataFrame([metric_data])
                    df.to_csv(os.path.join(csv_dir, f"{metric_name}.csv"), index=False)
                elif metric_name == "rouge":
                    # Rouge scores need special handling
                    rows = []
                    for rouge_type, scores in metric_data.items():
                        row = {"rouge_type": rouge_type}
                        row.update(scores)
                        rows.append(row)
                    if rows:
                        df = pd.DataFrame(rows)
                        df.to_csv(os.path.join(csv_dir, f"{metric_name}.csv"), index=False)
                else:
                    # Generic handling for other dictionary metrics
                    try:
                        df = pd.DataFrame([metric_data])
                        df.to_csv(os.path.join(csv_dir, f"{metric_name}.csv"), index=False)
                    except Exception as e:
                        logger.warning(f"Could not save {metric_name} as CSV: {e}")
    
    @staticmethod
    def _make_serializable(obj):
        """Make an object JSON-serializable."""
        if isinstance(obj, dict):
            return {k: ModelEvaluator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelEvaluator._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
