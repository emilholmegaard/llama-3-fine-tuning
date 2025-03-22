# Configuration Templates

This directory contains configuration templates for various fine-tuning scenarios. Use these templates as a starting point for your own fine-tuning experiments.

## Available Templates

- [**8B Model - Consumer GPU (16GB VRAM)**](./consumer_gpu_8b.yaml): For fine-tuning the 8B parameter model on a consumer GPU with 16GB VRAM
- [**8B Model - Low Memory (8GB VRAM)**](./low_memory_8b.yaml): For fine-tuning the 8B parameter model on a GPU with limited VRAM
- [**70B Model - Multi-GPU**](./multi_gpu_70b.yaml): For fine-tuning the 70B parameter model across multiple GPUs
- [**Instruction Fine-Tuning**](./instruction_tuning.yaml): Specialized configuration for instruction fine-tuning
- [**Document Processing**](./document_processing.yaml): Configuration for document-based fine-tuning

## How to Use

1. Copy the appropriate template to your working directory:
   ```bash
   cp config/templates/consumer_gpu_8b.yaml config/my_finetune_config.yaml
   ```

2. Modify the configuration file to match your specific requirements:
   ```bash
   nano config/my_finetune_config.yaml
   ```

3. Run fine-tuning with your configuration:
   ```bash
   python scripts/run_finetuning.py --config config/my_finetune_config.yaml
   ```

## Customizing Templates

When customizing these templates, focus on:

1. **Data paths**: Update paths to your specific dataset files
2. **Output directory**: Change the output directory for your fine-tuned model
3. **Training parameters**: Adjust batch size, learning rate, etc. based on your hardware and dataset
4. **LoRA parameters**: Modify LoRA settings based on your specific task

See the [Advanced Configuration Guide](../../docs/fine_tuning/advanced_configuration.md) for detailed explanations of all parameters.