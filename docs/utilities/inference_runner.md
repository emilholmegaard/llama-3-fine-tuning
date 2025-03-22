# Inference Runner

The `run_inference.py` script provides a flexible interface for running inference with fine-tuned Llama models. It supports running inference with single prompts, batch files, or in interactive mode.

## Features

- Run inference with fine-tuned Llama models
- Support for single prompts, batch processing, and interactive mode
- Track generation metrics (time, tokens/second)
- Load PEFT/LoRA models with automatic base model detection
- Configurable generation parameters (temperature, top-p, etc.)
- Various input and output formats (txt, csv, jsonl)

## Usage

```bash
# Single prompt
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --prompt "Analyze the following error log: ..." \
    --temperature 0.7 \
    --show_metrics

# Batch processing
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --input_file prompts.csv \
    --input_column prompt \
    --output_file responses.csv \
    --max_new_tokens 512

# Interactive mode
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --interactive \
    --save_session
```

## Arguments

### Model Options

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to fine-tuned model directory |
| `--quantize` | Load model in 4-bit precision |
| `--device` | Device to run inference on: auto, cuda, cpu (default: auto) |

### Input Options (choose one)

| Argument | Description |
|----------|-------------|
| `--prompt` | Single prompt for inference |
| `--input_file` | File containing prompts (txt, csv, jsonl) |
| `--interactive` | Run in interactive mode |

### File Options

| Argument | Description |
|----------|-------------|
| `--input_column` | Column name for prompts in CSV/JSONL input (default: prompt) |
| `--output_column` | Column name for responses in CSV/JSONL output (default: response) |

### Generation Options

| Argument | Description |
|----------|-------------|
| `--max_new_tokens` | Maximum number of tokens to generate (default: 256) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--top_p` | Nucleus sampling probability (default: 0.9) |
| `--top_k` | Top-k sampling (default: 50) |
| `--do_sample` | Use sampling instead of greedy decoding |
| `--num_beams` | Number of beams for beam search (default: 1) |

### Output Options

| Argument | Description |
|----------|-------------|
| `--output_file` | File to save inference results |
| `--output_format` | Output format: txt, csv, jsonl |
| `--show_metrics` | Show generation metrics (time, tokens/sec) |
| `--save_session` | Save the interactive session to a file |

## Examples

### Single Prompt Inference

Run inference with a single prompt:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --prompt "Summarize the following document: ..." \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --output_file single_response.txt
```

### Batch Processing from CSV

Process a CSV file with multiple prompts:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --input_file data/prompts/test_prompts.csv \
    --input_column query \
    --output_column answer \
    --output_format csv \
    --show_metrics
```

Sample input CSV:
```csv
id,query
1,"What is the main error in the log?"
2,"Analyze the following code snippet: ..."
3,"Summarize the differences between these log patterns"
```

### Batch Processing from JSONL

Process a JSONL file with multiple prompts:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --input_file data/prompts/test_prompts.jsonl \
    --output_format jsonl
```

Sample input JSONL:
```jsonl
{"prompt": "Explain the following error: ..."}
{"prompt": "What is causing this issue: ..."}
```

### Interactive Mode

Run the model in interactive mode:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --interactive \
    --temperature 0.8 \
    --max_new_tokens 1024 \
    --save_session
```

Interactive mode allows you to:
- Enter prompts directly
- Get immediate responses
- See generation metrics (if enabled)
- Save the entire conversation history

### Using a Quantized Model

Load the model in 4-bit precision for faster inference or to fit larger models on consumer GPUs:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --prompt "Analyze this log entry: ..." \
    --quantize
```

### Generation with Sampling

Use sampling with temperature and top-p for more diverse outputs:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --prompt "Generate creative solutions for: ..." \
    --temperature 0.9 \
    --top_p 0.92 \
    --do_sample
```

### Beam Search

Use beam search for potentially higher quality outputs:

```bash
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --prompt "What caused this error: ..." \
    --num_beams 5 \
    --max_new_tokens 512
```

## Output Formats

### Text (txt)

Plain text format with prompts and responses clearly marked:

```
Prompt 1:
What is the main error in this log?

Response 1:
The main error is a database connection timeout caused by...

--------------------------------
```

### CSV (csv)

Tabular format with prompt and response columns:

```csv
prompt,response
"What is the main error in this log?","The main error is a database connection timeout caused by..."
```

### JSONL (jsonl)

Each line contains a JSON object with prompt, response, and timestamp:

```jsonl
{"prompt": "What is the main error in this log?", "response": "The main error is a database connection timeout caused by...", "timestamp": "2025-03-22T08:30:45.123456"}
```

## Integration with Other Scripts

This utility is typically used after fine-tuning and evaluating models:

1. Fine-tune model: `run_finetuning.py`
2. Evaluate model: `evaluate_model.py`
3. Run inference: `run_inference.py`

Example workflow:

```bash
# Fine-tune model
python scripts/run_finetuning.py --config config/finetune_config.yaml

# Evaluate model
python scripts/evaluate_model.py \
    --model_path data/models/fine-tuned-model/ \
    --test_data data/processed/dataset/test.jsonl

# Run inference
python scripts/run_inference.py \
    --model_path data/models/fine-tuned-model/ \
    --interactive
```

## Tips for Effective Use

1. **Adjust Parameters**: Experiment with temperature, top-p, and top-k to find the right balance between diversity and coherence for your use case.

2. **Quantization**: Use the `--quantize` flag for larger models or when running on devices with limited memory.

3. **Batch Processing**: For large batch files, ensure you have sufficient memory. You can reduce `--max_new_tokens` if needed.

4. **Interactive Mode**: Use interactive mode during development to quickly test different prompts and see how the model responds.

5. **Generation Metrics**: Enable `--show_metrics` to get insights into model performance, especially useful when optimizing for speed.
