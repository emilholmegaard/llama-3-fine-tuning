# Hyperparameter Optimization

This guide explains how to perform hyperparameter optimization to find the best configuration for fine-tuning Llama 3.3 models on your specific dataset and task.

## Why Optimize Hyperparameters?

Hyperparameter optimization can significantly improve the performance of fine-tuned models by:

1. Finding the optimal learning rate, batch size, and other training parameters
2. Identifying the best LoRA/QLoRA configuration for your specific task
3. Maximizing performance within memory and time constraints
4. Preventing overfitting and ensuring good generalization

## Key Hyperparameters to Optimize

For Llama 3.3 fine-tuning, focus on these key hyperparameters:

### Training Parameters
- **Learning rate**: One of the most important parameters to optimize
- **Batch size**: Affects training stability and convergence speed
- **Training epochs**: Determines how long to train the model
- **Learning rate schedule**: Affects optimization dynamics

### LoRA Parameters
- **Rank (r)**: Controls the capacity of the LoRA adapters
- **Alpha**: Scaling factor for LoRA updates
- **Target modules**: Which layers to apply LoRA to

### Data Parameters
- **Max sequence length**: Affects memory usage and context window
- **Sampling strategy**: How to sample from the dataset

## Using the Hyperparameter Optimization Script

The repository includes a script for automated hyperparameter optimization:

```bash
python scripts/run_hyperparameter_optimization.py \
  --base_model meta-llama/Llama-3.3-8B \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --output_dir data/models/hpo-results/ \
  --num_trials 10 \
  --optimization_objective "eval_loss" \
  --search_space config/hpo_search_space.yaml
```

### Available Arguments

- `--base_model`: Base model to optimize
- `--train_file`: Training data file
- `--validation_file`: Validation data file
- `--output_dir`: Directory to save results
- `--num_trials`: Number of hyperparameter configurations to try
- `--optimization_objective`: Metric to optimize (e.g., "eval_loss", "eval_accuracy")
- `--search_space`: YAML file defining hyperparameter search space
- `--time_budget_hours`: Optional time limit for optimization (in hours)
- `--pruning_factor`: Early stopping factor for poorly performing trials
- `--parallel_trials`: Number of trials to run in parallel (requires multiple GPUs)

## Defining the Search Space

Create a search space YAML file with the ranges or options for each hyperparameter:

```yaml
# Example: config/hpo_search_space.yaml
training:
  learning_rate:
    type: "log_uniform"
    min: 1e-6
    max: 1e-4
  batch_size:
    type: "categorical"
    values: [1, 2, 4, 8]
  num_train_epochs:
    type: "int"
    min: 1
    max: 5
  warmup_ratio:
    type: "uniform"
    min: 0.03
    max: 0.1
  weight_decay:
    type: "log_uniform"
    min: 0.001
    max: 0.1

lora:
  r:
    type: "categorical"
    values: [8, 16, 32, 64]
  alpha:
    type: "categorical"
    values: [16, 32, 64, 128]
  dropout:
    type: "uniform"
    min: 0.0
    max: 0.2
```

### Search Space Types

- **log_uniform**: Samples from a log-uniform distribution (good for learning rates)
- **uniform**: Samples from a uniform distribution (good for dropout rates)
- **int**: Samples integer values (good for epochs or rank)
- **categorical**: Samples from a list of discrete values (good for batch sizes or model types)

## Understanding Optimization Results

After running the optimization, a report will be generated in the output directory:

```
data/models/hpo-results/
├── best_config.yaml        # Best hyperparameter configuration
├── trials_summary.csv      # Summary of all trials
├── optimization_plot.png   # Plot of optimization progress
└── trial_results/          # Individual trial results
    ├── trial_0/            # First trial results
    ├── trial_1/            # Second trial results
    └── ...
```

The script will also output the best configuration found:

```
Best hyperparameter configuration:
{
  "training": {
    "learning_rate": 2.35e-5,
    "batch_size": 4,
    "num_train_epochs": 3,
    "warmup_ratio": 0.063,
    "weight_decay": 0.023
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.052
  }
}

Best value: 0.723 (higher is better)
```

## Optimization Strategies

### Basic Strategy (Limited Resources)

For resource-constrained environments:

1. Start with a small search space focusing on learning rate and LoRA rank
2. Run a small number of trials (5-10)
3. Use early stopping to terminate poorly performing trials
4. Focus on shorter training runs for initial exploration

Example:
```bash
python scripts/run_hyperparameter_optimization.py \
  --base_model meta-llama/Llama-3.3-8B \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --output_dir data/models/hpo-basic/ \
  --num_trials 5 \
  --pruning_factor 2.0 \
  --search_space config/hpo_basic_search_space.yaml
```

### Comprehensive Strategy (More Resources)

With more compute resources available:

1. Define a broader search space with more hyperparameters
2. Run more trials (20-50)
3. Use parallel trials if multiple GPUs are available
4. Allocate more time for each trial to better evaluate performance

Example:
```bash
python scripts/run_hyperparameter_optimization.py \
  --base_model meta-llama/Llama-3.3-8B \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --output_dir data/models/hpo-comprehensive/ \
  --num_trials 30 \
  --parallel_trials 3 \
  --optimization_objective "eval_loss" \
  --search_space config/hpo_comprehensive_search_space.yaml
```

## Advanced Optimization Techniques

### Multi-objective Optimization

Optimize for multiple objectives simultaneously:

```bash
python scripts/run_hyperparameter_optimization.py \
  --base_model meta-llama/Llama-3.3-8B \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --output_dir data/models/hpo-multi-objective/ \
  --num_trials 20 \
  --optimization_objective "multi:eval_loss,eval_accuracy" \
  --optimization_weights "0.7,0.3" \
  --search_space config/hpo_search_space.yaml
```

### Memory-Constrained Optimization

Optimize within specific memory constraints:

```bash
python scripts/run_hyperparameter_optimization.py \
  --base_model meta-llama/Llama-3.3-8B \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --output_dir data/models/hpo-memory-constrained/ \
  --num_trials 15 \
  --memory_constraint 8 \
  --search_space config/hpo_memory_constrained_space.yaml
```

## Interpreting and Using Results

After optimization, analyze the results to understand hyperparameter importance:

1. **Parameter importance**: Identify which parameters had the most impact
2. **Correlation analysis**: Find relationships between parameters
3. **Learning curves**: Compare training dynamics across trials

Example parameter importance plot:
```
Parameter Importance:
1. learning_rate      (47.3%)
2. lora_r             (23.8%)
3. batch_size         (12.1%)
4. warmup_ratio       (8.7%)
5. weight_decay       (4.2%)
6. lora_alpha         (2.6%)
7. lora_dropout       (1.3%)
```

## Best Practices

1. **Start simple**: Begin with a focused search space and expand based on results
2. **Follow a process**:
   - Coarse search → Refined search → Final tuning
3. **Monitor carefully**:
   - Track both training and validation metrics
   - Look for signs of overfitting
4. **Save computation**:
   - Use early stopping for poor trials
   - Sample hyperparameters efficiently
5. **Be patient**:
   - Good hyperparameter optimization can take time
   - The performance improvements are often worth it

## Common Pitfalls

1. **Too narrow search ranges**: May miss optimal values outside your initial assumptions
2. **Too many parameters**: Increases search space complexity and time required
3. **Insufficient evaluation**: Short training runs may not reflect final performance
4. **Overfitting to validation set**: Hyperparameters may tune to validation quirks

## Next Steps

After finding optimal hyperparameters:

1. Train a final model with the best configuration
2. Validate on a separate test set
3. Consider ensemble methods for further improvements
4. Deploy and monitor the model's performance

For more information, refer to:
- [Basic Fine-Tuning Guide](./basic_fine_tuning.md)
- [Advanced Configuration Options](./advanced_configuration.md)
- [Memory Optimization Guide](../memory_optimization.md)