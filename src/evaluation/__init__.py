"""
Evaluation module for assessing the performance of fine-tuned Llama-3 models.

This module provides tools and utilities for evaluating language models with
a focus on perplexity, generation quality, and task-specific metrics.
"""

from src.evaluation.metrics import (
    calculate_perplexity,
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_exact_match,
    calculate_f1_score,
    calculate_accuracy
)

from src.evaluation.evaluator import ModelEvaluator

__all__ = [
    'calculate_perplexity',
    'calculate_rouge_scores',
    'calculate_bleu_score',
    'calculate_exact_match',
    'calculate_f1_score',
    'calculate_accuracy',
    'ModelEvaluator'
]
