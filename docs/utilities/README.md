# Utility Scripts

This directory contains documentation for the utility scripts used in the Llama 3 fine-tuning pipeline. These utilities help with data processing, model training, evaluation, and inference.

## Overview

The utilities provide a complete workflow for fine-tuning Llama 3 models:

1. **Data Processing**: Convert raw data to formats suitable for training
2. **Model Evaluation**: Assess model performance with advanced metrics
3. **Model Comparison**: Compare different model versions
4. **Inference**: Run inference with fine-tuned models
5. **Batch Processing**: Process documents and logs in parallel

## Available Utilities

| Utility | Description | Documentation |
|---------|-------------|---------------|
| `batch_processor.py` | Process batches of documents and logs | [Batch Processor](./batch_processor.md) |
| `prepare_dataset.py` | Convert processed data to training format | [Data Conversion](./data_conversion.md) |
| `run_finetuning.py` | Fine-tune Llama 3 models | [Fine-Tuning](../fine_tuning/basic_fine_tuning.md) |
| `evaluate_model.py` | Evaluate model performance | [Evaluation](../fine_tuning/evaluation.md) |
| `advanced_evaluate.py` | Advanced evaluation with detailed analysis | [Advanced Evaluation](./advanced_evaluation.md) |
| `compare_models.py` | Compare performance across model versions | [Model Comparison](./model_comparison.md) |
| `run_inference.py` | Run inference with fine-tuned models | [Inference Runner](./inference_runner.md) |

## Typical Workflow

A complete fine-tuning workflow using these utilities typically involves:

1. **Process Raw Data**
   ```bash
   python scripts/batch_processor.py --input_dirs data/raw/ --recursive
   ```

2. **Prepare Training Dataset**
   ```bash
   python scripts/prepare_dataset.py
   ```

3. **Fine-tune Model**
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml
   ```

4. **Evaluate Model**
   ```bash
   python scripts/evaluate_model.py \
       --model_path data/models/fine-tuned-model/ \
       --test_data data/processed/dataset/test.jsonl
   ```

5. **Run Inference**
   ```bash
   python scripts/run_inference.py \
       --model_path data/models/fine-tuned-model/ \
       --interactive
   ```

## Further Reading

- [Getting Started Guide](../getting_started.md)
- [Data Preparation Guide](../data_preparation.md)
- [Fine-Tuning Guides](../fine_tuning/README.md)
- [Memory Optimization Guide](../memory_optimization.md)
- [Troubleshooting Guide](../troubleshooting.md)
