# Basic Fine-Tuning Guide

This guide provides a step-by-step tutorial for fine-tuning Llama 3.3 models using the simplest approach. Follow these instructions to get started quickly with fine-tuning.

## Prerequisites

Before you begin:

1. Complete the [installation](../getting_started.md)
2. Prepare your dataset following the [data preparation guide](../data_preparation.md)
3. Ensure you have access to a GPU with at least 16GB of VRAM (for the 8B model)

## Overview of the Fine-Tuning Process

Fine-tuning consists of these steps:

1. Prepare your training data
2. Configure the training parameters
3. Run the fine-tuning process
4. Evaluate the results

## Step 1: Prepare Your Data

Ensure your data is in the correct format. For basic fine-tuning, your data should be in a JSONL format with the following structure:

For instruction/response format:
```json
{"instruction": "Your instruction here", "input": "Optional input here", "output": "Expected output here"}
```

For text completion:
```json
{"text": "Your training text here"}
```

Place your training data in `data/processed/dataset/train.jsonl` and validation data in `data/processed/dataset/val.jsonl`.

## Step 2: Configure Training Parameters

Create a basic configuration file or use the default one at `config/finetune_config.yaml`. For a basic setup, you can use the following configuration:

```yaml
model:
  base_model: "meta-llama/Llama-3.3-8B"
  output_dir: "data/models/finetuned-model/"

training:
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  max_steps: -1
  warmup_steps: 100
  save_steps: 500
  logging_steps: 50
  bf16: true

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

Adjust the parameters based on your specific requirements and hardware constraints.

## Step 3: Run Fine-Tuning

Execute the fine-tuning script with your configuration:

```bash
python scripts/run_finetuning.py --config config/finetune_config.yaml
```

If you're using a machine with limited GPU memory, consider using the memory-efficient training script instead:

```bash
python scripts/run_memory_efficient_training.py \
  --model_name_or_path meta-llama/Llama-3.3-8B \
  --output_dir data/models/finetuned-model/ \
  --train_file data/processed/dataset/train.jsonl \
  --validation_file data/processed/dataset/val.jsonl \
  --bits 4 \
  --lora_r 16 \
  --auto_find_batch_size
```

## Step 4: Monitor Training Progress

During training, you'll see output like this:

```
***** Running training *****
  Num examples = 1000
  Num Epochs = 3
  Instantaneous batch size per device = 4
  Total train batch size = 16
  Gradient Accumulation steps = 4
  Total optimization steps = 188
  
Step 50/188: loss=2.345, learning_rate=1.55e-05, epoch=0.8
Step 100/188: loss=1.879, learning_rate=1.22e-05, epoch=1.6
Step 150/188: loss=1.542, learning_rate=0.66e-05, epoch=2.4
Step 188/188: loss=1.349, learning_rate=0.00e+00, epoch=3.0
```

Training metrics will also be saved to the output directory, where you can monitor them using TensorBoard:

```bash
tensorboard --logdir data/models/finetuned-model/runs
```

## Step 5: Evaluate Your Model

After training completes, evaluate your model using the evaluation script:

```bash
python scripts/evaluate_model.py \
  --model_path data/models/finetuned-model/ \
  --test_data data/processed/dataset/test.jsonl \
  --output_dir data/evaluation/
```

This will generate evaluation metrics and sample outputs for your fine-tuned model.

## Step 6: Use Your Fine-Tuned Model

Once you're satisfied with the model's performance, you can use it for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the model
model_path = "data/models/finetuned-model/"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
response = pipe("Your prompt or instruction here", 
                max_length=200, 
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1)

print(response[0]["generated_text"])
```

## Tips for Success

- **Start small**: Begin with a small dataset and short training duration to test your setup
- **Validate your data**: Ensure your dataset is properly formatted and contains high-quality examples
- **Monitor loss values**: A healthy training run should show consistently decreasing loss values
- **Save checkpoints**: Use the `save_steps` parameter to save intermediate checkpoints
- **Experiment with learning rates**: Try different learning rates (e.g., 1e-5, 2e-5, 5e-5) to find the optimal value

## Troubleshooting

If you encounter issues:

- **Out of memory errors**: Reduce batch size, sequence length, or use the memory-efficient script
- **Poor performance**: Ensure your dataset is relevant to your task and contains enough examples
- **Training crashes**: Check your data format and reduce training parameters
- **Slow training**: Use mixed precision (bf16/fp16) and optimize batch size and gradient accumulation

## Next Steps

Once you've successfully completed a basic fine-tuning run, consider:

- Exploring [advanced configuration options](./advanced_configuration.md)
- Learning about [memory optimization techniques](../memory_optimization.md)
- Experimenting with [hyperparameter optimization](./hyperparameter_optimization.md) to improve performance