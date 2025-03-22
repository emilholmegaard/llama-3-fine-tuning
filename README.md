# Llama 3.3 Fine-Tuning

A Python project for fine-tuning Llama 3.3 models using document data from Word files and database logs. This tool is designed to help with legacy application carve-outs by training models on application-specific documentation and logs.

## Features

- **Word Document Processing**: Extract and process content from Word documents, preserving folder structure for categorization
- **Database Log Integration**: Parse and incorporate database logs as training data
- **Flexible Data Pipeline**: Configurable data processing steps for cleaning and formatting
- **Fine-Tuning Integration**: Support for Llama 3.3 fine-tuning using modern techniques
- **Evaluation Tools**: Metrics and evaluation scripts to assess model performance
- **Memory Optimization**: Advanced QLoRA techniques for fine-tuning in memory-constrained environments
- **Batch Processing**: Process large collections of documents and logs in parallel
- **Inference Utilities**: Run inference with fine-tuned models in various modes
- **Model Comparison**: Compare performance across different model versions

## Prerequisites

- Python 3.10+
- Git
- 16+ GB RAM for processing large documents
- GPU access for fine-tuning (optional but recommended)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/emilholmegaard/llama-3-fine-tuning.git
cd llama-3-fine-tuning
pip install -r requirements.txt
```

## Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

- [**Getting Started**](./docs/getting_started.md): Installation, setup, and prerequisites
- [**Data Preparation**](./docs/data_preparation.md): Processing documents and logs
- [**Fine-Tuning Guides**](./docs/fine_tuning/README.md):
  - [Basic Fine-Tuning](./docs/fine_tuning/basic_fine_tuning.md)
  - [Advanced Configuration](./docs/fine_tuning/advanced_configuration.md)
  - [Memory Optimization](./docs/memory_optimization.md)
  - [Hyperparameter Optimization](./docs/fine_tuning/hyperparameter_optimization.md)
- [**Utility Scripts**](./docs/utilities/README.md):
  - [Batch Processor](./docs/utilities/batch_processor.md)
  - [Data Conversion](./docs/utilities/data_conversion.md)
  - [Model Comparison](./docs/utilities/model_comparison.md)
  - [Inference Runner](./docs/utilities/inference_runner.md)
- [**Troubleshooting**](./docs/troubleshooting.md): Common issues and solutions

## Example Notebooks

Interactive Jupyter notebooks demonstrating various workflows are available in the [notebooks](./notebooks) directory:

- [Basic Fine-Tuning](./notebooks/01_basic_fine_tuning.ipynb): Step-by-step guide to fine-tuning with LoRA
- Additional notebooks for document processing, database logs, and advanced techniques

## Configuration Templates

Ready-to-use configuration templates for different hardware setups and use cases are available in [config/templates](./config/templates):

- [Consumer GPU (16GB VRAM)](./config/templates/consumer_gpu_8b.yaml)
- [Low Memory (8GB VRAM)](./config/templates/low_memory_8b.yaml)

## Project Structure

```
llama-3-fine-tuning/
├── config/                  # Configuration files
│   ├── default_config.yaml  # Default configuration
│   ├── finetune_config.yaml # Fine-tuning specific settings
│   └── templates/           # Configuration templates
├── data/                    # Data directory (git-ignored)
│   ├── raw/                 # Place raw data here
│   ├── processed/           # Processed data
│   └── models/              # Fine-tuned models
├── docs/                    # Documentation
│   ├── getting_started.md   # Installation and setup guide
│   ├── data_preparation.md  # Data processing guide
│   ├── fine_tuning/         # Fine-tuning guides
│   ├── utilities/           # Utility scripts documentation
│   └── memory_optimization.md # Memory optimization guide
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data processing modules
│   │   ├── word_processor.py  # Word document processing
│   │   ├── db_processor.py    # Database log processing
│   │   └── dataset.py         # Dataset creation
│   ├── models/              # Model-related code
│   │   ├── llama_wrapper.py   # Llama model wrapper
│   │   ├── fine_tuning.py     # Fine-tuning utilities
│   │   ├── qlora_optimizer.py # QLoRA optimization
│   │   └── memory_efficient_trainer.py # Memory-efficient training
│   ├── utils/               # Utility functions
│   │   ├── memory_utils.py    # Memory monitoring utilities
│   │   ├── memory_estimation.py # Memory estimation utilities
│   │   ├── memory_tracker.py   # Memory tracking utilities
│   │   ├── batch_optimizer.py  # Batch size optimization
│   │   └── cpu_offload.py      # CPU offloading utilities
│   └── evaluation/          # Evaluation metrics and tools
├── scripts/                 # Utility scripts
│   ├── process_docs.py      # Process Word documents
│   ├── process_logs.py      # Process DB logs
│   ├── prepare_dataset.py   # Prepare training dataset
│   ├── run_finetuning.py    # Run fine-tuning
│   ├── run_memory_efficient_training.py # Run memory-efficient training
│   ├── evaluate_model.py    # Evaluate model performance
│   ├── advanced_evaluate.py # Advanced evaluation tools
│   ├── compare_models.py    # Compare model versions
│   ├── run_inference.py     # Run inference with models
│   └── batch_processor.py   # Process documents and logs in batch
├── tests/                   # Unit tests
├── .gitignore               # Git ignore file
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Usage Guide

### 1. Data Preparation

#### Batch Processing

Process multiple documents and logs in parallel:

```bash
python scripts/batch_processor.py \
    --input_dirs data/raw/documents/ data/raw/logs/ \
    --output_dir data/processed/ \
    --recursive \
    --preserve_structure
```

For more options, see the [Batch Processor Documentation](./docs/utilities/batch_processor.md).

#### Processing Word Documents

Place your Word documents in `data/raw/documents/` maintaining any folder structure you want to preserve for categorization.

Run the document processing script:

```bash
python scripts/process_docs.py --input_dir data/raw/documents/ --output_dir data/processed/documents/ --recursive
```

Options:
- `--input_dir`: Directory containing Word documents
- `--output_dir`: Directory to save processed documents
- `--recursive`: Process documents in subdirectories
- `--preserve_structure`: Maintain folder structure in output

#### Processing Database Logs

Place your database logs in `data/raw/logs/`.

Run the log processing script:

```bash
python scripts/process_logs.py --input_dir data/raw/logs/ --output_dir data/processed/logs/ --format sql
```

Options:
- `--input_dir`: Directory containing log files
- `--output_dir`: Directory to save processed logs
- `--format`: Log format (sql, json, csv)

### 2. Dataset Preparation

Combine processed documents and logs into a training dataset:

```bash
python scripts/prepare_dataset.py --docs_dir data/processed/documents/ --logs_dir data/processed/logs/ --output_dir data/processed/dataset/ --train_split 0.8
```

Options:
- `--docs_dir`: Directory with processed documents
- `--logs_dir`: Directory with processed logs
- `--output_dir`: Directory to save the dataset
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--format`: Output format (jsonl, csv, parquet)

For more options, see the [Data Conversion Documentation](./docs/utilities/data_conversion.md).

### 3. Fine-Tuning

#### Standard Fine-Tuning

Run the standard fine-tuning script:

```bash
python scripts/run_finetuning.py --config config/finetune_config.yaml
```

The configuration file allows you to set:
- Model parameters (learning rate, batch size, etc.)
- Training duration (epochs, steps)
- Evaluation metrics
- Output paths

#### Memory-Efficient Fine-Tuning

For environments with limited GPU memory, use the memory-efficient training script:

```bash
python scripts/run_memory_efficient_training.py \
    --model_name_or_path meta-llama/Llama-3.3-8B \
    --output_dir ./outputs \
    --bits 4 \
    --lora_r 16 \
    --auto_find_batch_size \
    --auto_find_parameters \
    --memory_profiling
```

This script implements advanced QLoRA optimization techniques to minimize memory usage. See [Memory Optimization Guide](docs/memory_optimization.md) for more details.

### 4. Evaluation

#### Basic Evaluation

Evaluate the fine-tuned model:

```bash
python scripts/evaluate_model.py \
    --model_path data/models/finetuned-model/ \
    --test_data data/processed/dataset/test.jsonl \
    --output_dir data/evaluation/
```

#### Advanced Evaluation

Perform detailed evaluation with error analysis:

```bash
python scripts/advanced_evaluate.py \
    --model_paths data/models/finetuned-model/ \
    --test_data data/processed/dataset/test.jsonl \
    --output_dir data/evaluation/advanced/ \
    --error_analysis \
    --visualize
```

#### Compare Models

Compare performance across different model versions:

```bash
python scripts/compare_models.py \
    --eval_dirs data/evaluation/ \
    --output_dir data/comparisons/ \
    --visualize \
    --trend_analysis
```

For more options, see the [Model Comparison Documentation](./docs/utilities/model_comparison.md).

### 5. Inference

Run inference with your fine-tuned model:

```bash
# Interactive mode
python scripts/run_inference.py \
    --model_path data/models/finetuned-model/ \
    --interactive

# Batch processing
python scripts/run_inference.py \
    --model_path data/models/finetuned-model/ \
    --input_file data/queries.csv \
    --output_file data/responses.csv
```

For more options, see the [Inference Runner Documentation](./docs/utilities/inference_runner.md).

## Fine-Tuning Configuration

Edit `config/finetune_config.yaml` to customize fine-tuning parameters. For detailed parameter explanations, see the [Advanced Configuration Guide](./docs/fine_tuning/advanced_configuration.md).

## Memory-Efficient Training

This project implements advanced memory optimization techniques for fine-tuning large language models on limited hardware:

- **4-bit Quantization**: Reduces model memory footprint by 4x
- **QLoRA**: Parameter-efficient fine-tuning without quality loss
- **Gradient Checkpointing**: Trades compute for memory
- **CPU Offloading**: Offloads optimizer states to CPU
- **Automatic Batch Sizing**: Dynamically finds optimal batch size
- **Memory Profiling**: Tracks and analyzes memory usage

These optimizations make it possible to fine-tune models like Llama 3.3 8B on consumer GPUs with as little as 8GB VRAM. See the [Memory Optimization Guide](docs/memory_optimization.md) for detailed documentation.

## Tips for Effective Fine-Tuning

1. **Quality Data**: Focus on the quality of your training data. Well-structured, clean data will produce better results.

2. **Hardware Requirements**: Fine-tuning Llama 3.3 requires significant GPU resources. Consider:
   - Using smaller parameter versions for testing
   - Using parameter-efficient fine-tuning methods (LoRA, QLoRA)
   - Cloud GPU services if local resources are insufficient

3. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and LoRA parameters.

4. **Evaluation Strategy**: Define clear evaluation metrics aligned with your application carve-out goals.

5. **Incremental Approach**: Start with a small dataset and gradually increase size and complexity.

## Troubleshooting

For a comprehensive list of common issues and solutions, see the [Troubleshooting Guide](./docs/troubleshooting.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
