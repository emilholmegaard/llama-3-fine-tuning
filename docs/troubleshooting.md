# Troubleshooting Guide

This guide addresses common issues encountered during the Llama 3.3 fine-tuning process, including memory problems, training errors, and performance issues.

## Memory Issues

### Out of Memory (OOM) Errors

```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 15.78 GiB total capacity; 12.56 GiB already allocated; 1.97 GiB free; 14.34 GiB reserved in total by PyTorch)
```

#### Possible Solutions

1. **Reduce batch size**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.batch_size 1
   ```

2. **Use gradient accumulation**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.gradient_accumulation_steps 16
   ```

3. **Reduce sequence length**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --data.max_seq_length 1024
   ```

4. **Use QLoRA**:
   ```bash
   python scripts/run_memory_efficient_training.py --model_name_or_path meta-llama/Llama-3.3-8B --bits 4
   ```

5. **Enable gradient checkpointing**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.gradient_checkpointing true
   ```

6. **Free up GPU memory**:
   - Close other applications using the GPU
   - Use `nvidia-smi` to check what's using your GPU memory
   - Restart your system if necessary

### CPU Memory Issues

```
MemoryError: Unable to allocate array with shape (3072, 4096) and data type float32
```

#### Possible Solutions

1. **Reduce preprocessing workers**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --data.preprocessing_num_workers 1
   ```

2. **Process data in smaller batches**:
   ```bash
   python scripts/process_docs.py --input_dir data/raw/documents/ --output_dir data/processed/documents/ --batch_size 10
   ```

3. **Use memory-mapped datasets**:
   ```bash
   python scripts/prepare_dataset.py --use_memory_mapping --output_format arrow
   ```

## Training Issues

### Loss Not Decreasing

#### Possible Solutions

1. **Adjust learning rate**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.learning_rate 5e-6
   ```

2. **Check data quality**:
   - Ensure data is properly formatted
   - Verify there are no corrupted samples
   - Check token lengths and distributions

3. **Try different optimizer settings**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.optim adamw_torch --training.weight_decay 0.01
   ```

4. **Use learning rate scheduler**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.lr_scheduler_type cosine --training.warmup_ratio 0.05
   ```

### Training Too Slow

#### Possible Solutions

1. **Use mixed precision training**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.bf16 true
   ```

2. **Increase batch size (if memory allows)**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.batch_size 8
   ```

3. **Use Flash Attention**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --model.use_flash_attention true
   ```

4. **Optimize data loading**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --data.preprocessing_num_workers 8
   ```

### Convergence Issues

#### Possible Solutions

1. **Train longer**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.num_train_epochs 5
   ```

2. **Adjust LoRA parameters**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --lora.r 32 --lora.alpha 64
   ```

3. **Check data relevance**:
   - Ensure your training data is relevant to your target task
   - Verify data is properly tokenized and processed

### Unexpected Errors

#### Common Error Messages and Solutions

1. **"Expected all tensors to be on the same device"**:
   - Ensure all tensors are on the same device (CPU or GPU)
   - Check for mixed device operations in custom code

2. **"IndexError: index out of range in self"**:
   - Check for incompatible tensor dimensions
   - Verify batch size and sequence length settings

3. **"RuntimeError: expected scalar type X but found scalar type Y"**:
   - Ensure consistent data types in your inputs
   - Check dtype settings in configuration

4. **"ValueError: Checkpoint version X not supported for XX"**:
   - Use compatible versions of transformers and peft libraries
   - Check if checkpoint was saved with a newer library version

## Model Quality Issues

### Poor Generated Text Quality

#### Possible Solutions

1. **Check fine-tuning data quality**:
   - Ensure high-quality, relevant training examples
   - Remove noisy or irrelevant samples

2. **Increase model capacity**:
   - Use larger LoRA rank:
     ```bash
     python scripts/run_finetuning.py --config config/finetune_config.yaml --lora.r 64 --lora.alpha 128
     ```
   - Fine-tune more layers:
     ```bash
     python scripts/run_finetuning.py --config config/finetune_config.yaml --lora.target_modules ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
     ```

3. **Adjust generation parameters**:
   - Try different temperature, top_p, and top_k values
   - Experiment with repetition penalties

### Overfitting

#### Possible Solutions

1. **Add regularization**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.weight_decay 0.05 --lora.dropout 0.1
   ```

2. **Early stopping**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.early_stopping true --training.early_stopping_patience 3
   ```

3. **Reduce training time**:
   ```bash
   python scripts/run_finetuning.py --config config/finetune_config.yaml --training.num_train_epochs 2
   ```

4. **Data augmentation**:
   - Increase dataset diversity
   - Implement text augmentation techniques

### Model Hallucinations

#### Possible Solutions

1. **Improve training data accuracy**:
   - Verify factual correctness of training examples
   - Remove ambiguous or incorrect information

2. **Add fact-checking examples**:
   - Include examples that demonstrate fact verification
   - Train on examples of when to say "I don't know"

3. **Adjust decoding parameters**:
   - Lower temperature (e.g., 0.3-0.7)
   - Increase presence_penalty to discourage repetition

## Installation and Environment Issues

### CUDA/GPU Issues

```
ImportError: cannot import name 'cuda' from 'torch'
```

or

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

#### Possible Solutions

1. **Check CUDA installation**:
   ```bash
   nvidia-smi
   ```

2. **Reinstall PyTorch with correct CUDA version**:
   ```bash
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA paths**:
   ```bash
   echo $LD_LIBRARY_PATH
   echo $CUDA_HOME
   ```

4. **Update GPU drivers**:
   - Download latest drivers from NVIDIA website
   - Follow installation instructions for your OS

### Library Version Conflicts

```
ImportError: cannot import name 'XXX' from 'transformers'
```

#### Possible Solutions

1. **Check and fix library versions**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

2. **Create a fresh virtual environment**:
   ```bash
   python -m venv new_env
   source new_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Check for conflicting packages**:
   ```bash
   pip check
   ```

## Filesystem and Data Issues

### Data Loading Errors

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/dataset/train.jsonl'
```

#### Possible Solutions

1. **Check file paths**:
   - Verify that files exist at the specified locations
   - Ensure correct permissions

2. **Create missing directories**:
   ```bash
   mkdir -p data/processed/dataset/
   ```

3. **Check file format compatibility**:
   - Ensure files are in the correct format
   - Validate JSON/JSONL files

### Permission Errors

```
PermissionError: [Errno 13] Permission denied: 'data/models/finetuned-model/'
```

#### Possible Solutions

1. **Fix permissions**:
   ```bash
   chmod -R 755 data/
   ```

2. **Run with appropriate user privileges**:
   - Use sudo if necessary (not recommended for Python)
   - Change ownership of directories

## Deployment Issues

### Model Loading Errors

```
OSError: Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF checkpoint, please set from_tf=True
```

#### Possible Solutions

1. **Check model format compatibility**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   # For LoRA models
   from peft import PeftModel, PeftConfig
   
   config = PeftConfig.from_pretrained("data/models/finetuned-model/")
   model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
   model = PeftModel.from_pretrained(model, "data/models/finetuned-model/")
   ```

2. **Convert adapter format if needed**:
   ```bash
   python scripts/convert_adapter.py --input_model data/models/finetuned-model/ --output_model data/models/converted-model/
   ```

### Inference Performance Issues

#### Possible Solutions

1. **Optimize model for inference**:
   ```bash
   python scripts/optimize_for_inference.py --model_path data/models/finetuned-model/ --output_path data/models/optimized-model/
   ```

2. **Use quantization for deployment**:
   ```bash
   python scripts/quantize_model.py --model_path data/models/finetuned-model/ --output_path data/models/quantized-model/ --bits 8
   ```

3. **Batch inputs for throughput**:
   - Process multiple inputs in a single forward pass
   - Use proper batching in your inference code

## Getting More Help

If you're still experiencing issues:

1. **Check the GitHub repository issues**:
   - Search existing issues for similar problems
   - Open a new issue with detailed information

2. **Provide diagnostic information**:
   - Include full error messages
   - Share your configuration file
   - Describe your environment (OS, GPU, CUDA version)
   - Include steps to reproduce the issue

3. **Try minimal reproduction**:
   - Create a simplified version of your code that demonstrates the issue
   - Test with a small subset of your data

4. **Community resources**:
   - Check Hugging Face forums
   - Search related discussions on Stack Overflow
   - Join relevant Discord communities