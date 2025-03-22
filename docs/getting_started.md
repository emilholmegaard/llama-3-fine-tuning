# Getting Started with Llama 3.3 Fine-Tuning

This guide will help you set up the environment and get started with fine-tuning Llama 3.3 models.

## Prerequisites

Before you begin, ensure you have the following:

### Hardware Requirements

- **RAM**: Minimum 16GB, recommended 32GB+ for processing large datasets
- **Storage**: At least 100GB free space for models and datasets
- **GPU**: 
  - For 8B model: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
  - For 70B model: Multiple high-end GPUs or cloud GPU access
  - CPU-only setup is possible but extremely slow

### Software Requirements

- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Git**: For cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/emilholmegaard/llama-3-fine-tuning.git
cd llama-3-fine-tuning
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n llama-ft python=3.10
conda activate llama-ft
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, you may need to install PyTorch with the appropriate CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify Installation

Run the following command to verify that the installation was successful:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

## Repository Structure

The repository is organized as follows:

```
llama-3-fine-tuning/
├── config/                  # Configuration files
├── data/                    # Data directory (git-ignored)
├── docs/                    # Documentation (you are here)
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── requirements.txt         # Project dependencies
└── README.md                # Project overview
```

### Key Components

- **config/**: Contains configuration files for fine-tuning and data processing.
- **src/**: Contains the core implementation of the fine-tuning pipeline.
- **scripts/**: Contains utility scripts for data processing, fine-tuning, and evaluation.
- **notebooks/**: Contains example Jupyter notebooks for interactive exploration.

## Next Steps

Once you have set up the repository, you can:

1. [Prepare your training data](./data_preparation.md)
2. [Run a basic fine-tuning job](./fine_tuning/basic_fine_tuning.md)
3. [Evaluate your fine-tuned model](./evaluation/README.md)

## Troubleshooting

If you encounter issues during installation:

1. **CUDA compatibility issues**: Ensure PyTorch is installed with the correct CUDA version
2. **Memory errors during import**: Try importing only the necessary modules
3. **Library not found errors**: Check if all dependencies are installed correctly

For more detailed troubleshooting, see the [Troubleshooting](./troubleshooting.md) guide.