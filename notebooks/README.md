# Example Notebooks

This directory contains Jupyter notebooks that demonstrate various fine-tuning workflows and use cases for Llama 3.3 models.

## Basic Tutorials

- [**Basic Fine-Tuning**](./01_basic_fine_tuning.ipynb): A step-by-step guide to fine-tuning Llama 3.3 with LoRA
- [**Memory-Efficient Fine-Tuning**](./02_memory_efficient_fine_tuning.ipynb): Fine-tuning on consumer hardware with QLoRA
- [**Evaluation and Testing**](./03_evaluation_and_testing.ipynb): Techniques for evaluating fine-tuned models

## Document-Based Fine-Tuning

- [**Word Document Processing**](./04_word_document_processing.ipynb): Processing and fine-tuning on Word documents
- [**PDF Document Processing**](./05_pdf_document_processing.ipynb): Processing and fine-tuning on PDF documents

## Database Log Fine-Tuning

- [**SQL Log Processing**](./06_sql_log_processing.ipynb): Processing and fine-tuning on SQL database logs
- [**Transaction Log Fine-Tuning**](./07_transaction_log_fine_tuning.ipynb): Fine-tuning on transaction logs

## Advanced Workflows

- [**Hyperparameter Optimization**](./08_hyperparameter_optimization.ipynb): Finding optimal hyperparameters
- [**Multi-GPU Training**](./09_multi_gpu_training.ipynb): Distributed training across multiple GPUs
- [**Model Merging**](./10_model_merging.ipynb): Techniques for merging multiple fine-tuned models

## Usage Instructions

1. **Environment Setup**

   To run these notebooks, you'll need to install Jupyter:

   ```bash
   pip install jupyter
   ```

   Ensure you have all required dependencies installed:

   ```bash
   pip install -r ../requirements.txt
   ```

2. **Starting Jupyter**

   ```bash
   cd llama-3-fine-tuning
   jupyter notebook
   ```

   Navigate to the `notebooks` directory and open any notebook.

3. **Using Notebooks with GPU**

   To ensure notebooks can access your GPU, you may need to install a kernel with GPU support:

   ```bash
   python -m ipykernel install --user --name=llama-ft --display-name="Python (Llama Fine-Tuning)"
   ```

   Then select this kernel when running the notebooks.

4. **Notebook Dependencies**

   Some notebooks may require additional dependencies. These will be noted at the beginning of each notebook and can be installed directly from the notebook.

## Data Requirements

Most notebooks expect data in the standard directory structure:

```
data/
├── raw/
│   ├── documents/      # Raw document files
│   └── logs/           # Raw log files
├── processed/
│   ├── documents/      # Processed document data
│   ├── logs/           # Processed log data
│   └── dataset/        # Combined training datasets
└── models/             # Fine-tuned model outputs
```

Each notebook will provide instructions for preparing the necessary data.

## Customizing Notebooks

These notebooks are designed as starting points for your own fine-tuning projects. Feel free to modify them to fit your specific use case by:

1. Changing model parameters
2. Adapting data processing steps
3. Modifying training configurations
4. Adding custom evaluation metrics

## Troubleshooting

If you encounter issues running these notebooks:

1. Verify GPU accessibility:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0))
   ```

2. Check memory usage:
   ```python
   import torch
   print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
   print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
   ```

3. For more detailed troubleshooting, refer to the [Troubleshooting Guide](../docs/troubleshooting.md)