"""
Metrics for evaluating fine-tuned language models.

This module contains functions for calculating various evaluation metrics
for language models, including perplexity, ROUGE scores, BLEU scores,
and task-specific metrics.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Union, Tuple, Optional, Any
from datasets import Dataset
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    device: str = "cuda",
    max_length: int = 1024
) -> Dict[str, float]:
    """
    Calculate perplexity for a list of texts.

    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        texts: List of text sequences to evaluate
        batch_size: Batch size for processing
        device: Device to run calculations on ('cuda' or 'cpu')
        max_length: Maximum sequence length for tokenization

    Returns:
        Dictionary with perplexity score and token-level stats
    """
    model.eval()
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    model.to(device)
    
    # Process in batches
    all_loss = []
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize with padding
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = batch_encodings.input_ids.to(device)
            attention_mask = batch_encodings.attention_mask.to(device)
            
            # Prepare labels (shift input_ids for causal LM)
            labels = input_ids.clone()
            
            # Calculate loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            all_loss.append(loss.item())
            batch_tokens = attention_mask.sum().item()
            total_tokens += batch_tokens
    
    # Calculate perplexity from average loss
    avg_loss = sum(all_loss) / len(all_loss)
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens
    }


def calculate_rouge_scores(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores for predicted texts against references.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        rouge_types: Types of ROUGE metrics to calculate

    Returns:
        Dictionary with ROUGE scores
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=True
    )
    
    # Organize results
    scores = {}
    for rouge_type in rouge_types:
        scores[rouge_type] = {
            "precision": results[f"{rouge_type}_precision"],
            "recall": results[f"{rouge_type}_recall"],
            "fmeasure": results[f"{rouge_type}_fmeasure"]
        }
    
    return scores


def calculate_bleu_score(
    predictions: List[str],
    references: List[List[str]],
    max_ngram_order: int = 4
) -> Dict[str, float]:
    """
    Calculate BLEU scores for predicted texts against references.

    Args:
        predictions: List of predicted texts
        references: List of reference texts (can be multiple per prediction)
        max_ngram_order: Maximum n-gram order for BLEU calculation

    Returns:
        Dictionary with BLEU scores
    """
    sacrebleu = load("sacrebleu")
    results = sacrebleu.compute(
        predictions=predictions,
        references=references
    )
    
    return {
        "bleu": results["score"] / 100,  # Normalize to 0-1 range
        "precisions": [p / 100 for p in results["precisions"]]
    }


def calculate_exact_match(
    predictions: List[str],
    references: List[str],
    normalize: bool = True
) -> float:
    """
    Calculate exact match score (useful for QA tasks).

    Args:
        predictions: List of predicted answers
        references: List of reference answers
        normalize: Whether to normalize text before comparison

    Returns:
        Exact match score (0.0 to 1.0)
    """
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    
    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    return matches / len(predictions)


def calculate_f1_score(
    predictions: List[Union[str, int]],
    references: List[Union[str, int]],
    average: str = "macro",
    normalize_text_inputs: bool = False
) -> Dict[str, float]:
    """
    Calculate F1 score for classification or token-level tasks.

    Args:
        predictions: List of predicted labels or texts
        references: List of reference labels or texts
        average: Averaging method ('macro', 'micro', 'weighted')
        normalize_text_inputs: Whether to tokenize and normalize text inputs

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if normalize_text_inputs and isinstance(predictions[0], str):
        # Convert text to token sets for token-level F1
        pred_tokens = [set(normalize_text(p).split()) for p in predictions]
        ref_tokens = [set(normalize_text(r).split()) for r in references]
        
        # Calculate token-level F1
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for pred, ref in zip(pred_tokens, ref_tokens):
            if not pred and not ref:
                precision_scores.append(1.0)
                recall_scores.append(1.0)
                f1_scores.append(1.0)
            elif not pred or not ref:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)
            else:
                common = pred.intersection(ref)
                p = len(common) / len(pred) if pred else 0
                r = len(common) / len(ref) if ref else 0
                f1 = 2 * p * r / (p + r) if (p + r) else 0
                precision_scores.append(p)
                recall_scores.append(r)
                f1_scores.append(f1)
        
        return {
            "precision": sum(precision_scores) / len(precision_scores),
            "recall": sum(recall_scores) / len(recall_scores),
            "f1": sum(f1_scores) / len(f1_scores)
        }
    else:
        # Use sklearn for label-based F1
        precision = precision_score(references, predictions, average=average, zero_division=0)
        recall = recall_score(references, predictions, average=average, zero_division=0)
        f1 = f1_score(references, predictions, average=average, zero_division=0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


def calculate_accuracy(
    predictions: List[Union[str, int]], 
    references: List[Union[str, int]],
    normalize_text_inputs: bool = False
) -> float:
    """
    Calculate accuracy for classification tasks.

    Args:
        predictions: List of predicted labels or texts
        references: List of reference labels or texts
        normalize_text_inputs: Whether to normalize text inputs

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if normalize_text_inputs and isinstance(predictions[0], str):
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]
    
    return accuracy_score(references, predictions)


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent comparison.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for words with apostrophes
    import re
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_domain_specific_metrics(
    predictions: List[str],
    references: List[str],
    domain_type: str,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate domain-specific metrics based on the use case.
    Extend this function for custom domain-specific evaluation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        domain_type: Type of domain ('qa', 'summarization', 'classification', etc.)
        **kwargs: Additional domain-specific parameters
        
    Returns:
        Dictionary with domain-specific metrics
    """
    results = {}
    
    if domain_type == "qa":
        # Question answering metrics
        results["exact_match"] = calculate_exact_match(predictions, references)
        results.update(calculate_f1_score(predictions, references, normalize_text_inputs=True))
        
    elif domain_type == "summarization":
        # Summarization metrics
        rouge_results = calculate_rouge_scores(predictions, references)
        for rouge_type, scores in rouge_results.items():
            results[f"{rouge_type}_f1"] = scores["fmeasure"]
        
        if len(predictions) > 0 and len(references) > 0 and isinstance(references[0], str):
            # Add reference lists for BLEU
            references_for_bleu = [[r] for r in references]
            bleu_results = calculate_bleu_score(predictions, references_for_bleu)
            results["bleu"] = bleu_results["bleu"]
    
    elif domain_type == "classification":
        # Classification metrics
        results["accuracy"] = calculate_accuracy(predictions, references)
        results.update(calculate_f1_score(predictions, references))
        
    # Add more domain types as needed
    
    return results
