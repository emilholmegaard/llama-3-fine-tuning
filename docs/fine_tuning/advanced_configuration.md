# Advanced Configuration Options

This guide explains the advanced configuration options for fine-tuning Llama 3.3 models, providing detailed explanations of parameters and recommendations for different use cases.

## Configuration File Structure

The fine-tuning configuration file (typically `config/finetune_config.yaml`) contains several sections that control different aspects of the training process:

```yaml
model:  # Model configuration
  base_model: "meta-llama/Llama-3.3-8B"
  output_dir: "data/models/finetuned-model/"

training:  # Training process parameters
  learning_rate: 2e-5
  batch_size: 4
  # More training parameters...

data:  # Dataset configuration
  train_file: "data/processed/dataset/train.jsonl"
  # More data parameters...

lora:  # LoRA adaptation parameters
  use_lora: true
  r: 16
  # More LoRA parameters...

qlora:  # QLoRA configuration (optional)
  use_qlora: false
  bits: 4
  # More QLoRA parameters...
```

## Model Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `base_model` | The base model to fine-tune | `"meta-llama/Llama-3.3-8B"`, `"meta-llama/Llama-3.3-70B"` |
| `output_dir` | Directory to save the fine-tuned model | `"data/models/your-model-name/"` |
| `torch_dtype` | Precision for model computation | `"auto"`, `"bfloat16"`, `"float16"`, `"float32"` |
| `trust_remote_code` | Whether to trust code from the model's repository | `true`, `false` |
| `use_flash_attention` | Whether to use Flash Attention for faster training | `true`, `false` |
| `cache_dir` | Directory to cache downloaded models | `"data/cache/"` |

## Training Parameters

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `learning_rate` | Rate at which weights are updated | 1e-5 to 5e-5 | Smaller for larger models |
| `batch_size` | Number of examples per batch | 1 to 8 | Depends on GPU memory |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients | 1 to 16 | Increases effective batch size |
| `num_train_epochs` | Number of training epochs | 1 to 5 | More for small datasets |
| `max_steps` | Maximum number of training steps | -1 (all epochs) or specific count | Overrides epochs if set |
| `warmup_steps` | Steps for learning rate warmup | 100 to 500 | 5-10% of total steps |
| `warmup_ratio` | Ratio of warmup steps to total steps | 0.03 to 0.1 | Alternative to fixed steps |
| `weight_decay` | L2 regularization strength | 0.01 to 0.1 | Prevents overfitting |
| `lr_scheduler_type` | Learning rate schedule type | `"cosine"`, `"linear"`, `"constant"` | Cosine is often best |
| `logging_steps` | Frequency of logging | 10 to 100 | Lower for smaller datasets |
| `save_steps` | Frequency of checkpoint saving | 100 to 1000 | Memory vs. safety tradeoff |
| `save_total_limit` | Maximum number of checkpoints to keep | 1 to 5 | Limits disk usage |
| `fp16` | Whether to use 16-bit precision | `true`, `false` | For older GPUs |
| `bf16` | Whether to use bfloat16 precision | `true`, `false` | For newer GPUs |
| `optim` | Optimizer to use | `"adamw_torch"`, `"adamw_8bit"` | 8-bit saves memory |
| `adam_beta1` | First momentum coefficient | 0.9 | Default is usually fine |
| `adam_beta2` | Second momentum coefficient | 0.999 | Default is usually fine |
| `adam_epsilon` | Small constant for numerical stability | 1e-8 | Default is usually fine |
| `max_grad_norm` | Gradient clipping threshold | 1.0 | Prevents exploding gradients |
| `seed` | Random seed for reproducibility | Any integer | Ensures repeatable results |

## Data Parameters

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `train_file` | Path to training data | `"data/processed/dataset/train.jsonl"` | |
| `validation_file` | Path to validation data | `"data/processed/dataset/val.jsonl"` | |
| `max_seq_length` | Maximum sequence length | 512 to 4096 | Depends on model and GPU memory |
| `preprocessing_num_workers` | Number of workers for preprocessing | 4 to 16 | Depends on CPU cores |
| `overwrite_cache` | Whether to recompute cached files | `true`, `false` | Set true after data changes |
| `pad_to_max_length` | Whether to pad all sequences | `true`, `false` | true increases batch homogeneity |
| `data_format` | Format of input data | `"instruction"`, `"text"`, `"custom"` | Depends on task |
| `prompt_template` | Template for formatting prompts | Various templates | Depends on model and task |

## LoRA Parameters

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `use_lora` | Whether to use LoRA | `true`, `false` | true for memory efficiency |
| `r` | Rank of the update matrices | 8 to 64 | Higher = more capacity but more memory |
| `alpha` | Scaling factor for LoRA | 16 to 64 | Typically 2 × r |
| `dropout` | Dropout probability for LoRA | 0.05 to 0.1 | Prevents overfitting |
| `target_modules` | Which modules to apply LoRA to | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Model-specific |
| `bias` | Whether to train biases | `"none"`, `"all"`, `"lora_only"` | Usually "none" or "all" |
| `task_type` | Type of task | `"CAUSAL_LM"`, `"SEQ_CLS"` | Usually "CAUSAL_LM" for LLMs |

## QLoRA Parameters

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `use_qlora` | Whether to use QLoRA | `true`, `false` | true for extreme memory efficiency |
| `bits` | Quantization bits | 4, 8 | 4 for more memory savings |
| `double_quant` | Whether to use double quantization | `true`, `false` | true for more memory savings |
| `quant_type` | Quantization type | `"nf4"`, `"fp4"` | "nf4" is typically better |
| `use_4bit` | Whether to use 4-bit precision | `true`, `false` | Shorthand for bits=4 |
| `compute_dtype` | Computation precision | `"float16"`, `"bfloat16"` | bfloat16 for newer GPUs |

## Memory Optimization Parameters

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `gradient_checkpointing` | Whether to use gradient checkpointing | `true`, `false` | Trades compute for memory |
| `gradient_checkpointing_kwargs` | Additional checkpointing args | `{"use_reentrant": false}` | Model-dependent |
| `use_peft_gpu_offload` | Whether to offload LoRA adapters to CPU | `true`, `false` | Saves GPU memory |
| `auto_find_batch_size` | Automatically find largest working batch size | `true`, `false` | Convenient but may have trial runs |
| `auto_find_parameters` | Auto-configure memory parameters | `true`, `false` | Recommended for beginners |
| `memory_profiling` | Whether to profile memory usage | `true`, `false` | Useful for optimization |

## Example Configurations

### Basic LoRA Fine-Tuning (8B Model, 16GB+ GPU)

```yaml
model:
  base_model: "meta-llama/Llama-3.3-8B"
  output_dir: "data/models/llama-3-lora-basic/"

training:
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  warmup_steps: 100
  lr_scheduler_type: "cosine"
  bf16: true
  optim: "adamw_torch"

data:
  train_file: "data/processed/dataset/train.jsonl"
  validation_file: "data/processed/dataset/val.jsonl"
  max_seq_length: 2048
  preprocessing_num_workers: 4

lora:
  use_lora: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Memory-Efficient QLoRA (8B Model, 8GB GPU)

```yaml
model:
  base_model: "meta-llama/Llama-3.3-8B"
  output_dir: "data/models/llama-3-qlora-efficient/"

training:
  learning_rate: 1e-4
  batch_size: 1
  gradient_accumulation_steps: 16
  num_train_epochs: 3
  max_steps: -1
  warmup_ratio: 0.05
  logging_steps: 10
  save_steps: 100
  save_total_limit: 3
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
  auto_find_batch_size: true
  memory_profiling: true

data:
  train_file: "data/processed/dataset/train.jsonl"
  validation_file: "data/processed/dataset/val.jsonl"
  max_seq_length: 1024
  preprocessing_num_workers: 4

lora:
  use_lora: true
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

qlora:
  use_qlora: true
  bits: 4
  double_quant: true
  quant_type: "nf4"
```

### Large-Scale Fine-Tuning (70B Model, Multiple GPUs)

```yaml
model:
  base_model: "meta-llama/Llama-3.3-70B"
  output_dir: "data/models/llama-3-70b-tuned/"
  use_flash_attention: true

training:
  learning_rate: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 32
  num_train_epochs: 2
  warmup_ratio: 0.03
  logging_steps: 5
  save_steps: 200
  save_total_limit: 2
  bf16: true
  optim: "adamw_8bit"
  gradient_checkpointing: true

data:
  train_file: "data/processed/dataset/train.jsonl"
  validation_file: "data/processed/dataset/val.jsonl"
  max_seq_length: 2048
  preprocessing_num_workers: 16

lora:
  use_lora: true
  r: 32
  alpha: 64
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Tips for Optimizing Configuration

1. **Start with a baseline**: Begin with a standard configuration and adjust parameters incrementally
2. **Memory vs. speed trade-offs**:
   - Lower batch size: Uses less memory but slows training
   - Gradient checkpointing: Saves memory but increases computation time
   - Gradient accumulation: Simulates larger batch sizes with lower memory
3. **Prevent overfitting**:
   - Increase dropout values (0.05 to 0.1)
   - Add weight decay (0.01 to 0.1)
   - Reduce training epochs
4. **Monitor validation loss**:
   - Stop training when validation loss plateaus
   - Use early stopping to prevent overfitting
5. **Hardware-specific optimizations**:
   - For A100, H100 GPUs: Use bf16 precision
   - For older GPUs: Use fp16 precision
   - For low-memory GPUs: Use QLoRA with 4-bit quantization

## Debugging Configuration Issues

Common problems and solutions:

1. **Out of memory errors**: 
   - Reduce batch size
   - Enable gradient checkpointing
   - Use LoRA/QLoRA
   - Reduce model size or sequence length

2. **Slow training**: 
   - Increase batch size (if memory allows)
   - Use flash attention
   - Optimize preprocessing_num_workers
   - Use mixed precision training

3. **Poor convergence**:
   - Adjust learning rate (try 5e-6 to 5e-5)
   - Increase training epochs
   - Check data quality
   - Use cosine learning rate scheduler

4. **Overfitting**:
   - Add dropout
   - Increase weight decay
   - Reduce model capacity (lower LoRA rank)
   - Add more training data

## Further Information

For more in-depth discussions of specific parameters and techniques, refer to:
- [Memory Optimization Guide](../memory_optimization.md)
- [Hyperparameter Optimization](./hyperparameter_optimization.md)
