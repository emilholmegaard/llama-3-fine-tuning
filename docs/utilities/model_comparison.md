# Model Comparison Utility

The `compare_models.py` script allows you to compare metrics across different model versions without rerunning evaluations. It analyzes existing evaluation results, generates visualizations, and produces comprehensive reports.

## Features

- Compares evaluation metrics across multiple model versions
- Generates visualizations of model performance
- Creates trend analysis to track progress over time
- Ranks models based on multiple metrics
- Produces detailed reports in multiple formats

## Usage

```bash
python scripts/compare_models.py \
    --eval_dirs data/evaluation/ \
    --output_dir data/comparisons/ \
    --metrics perplexity rouge bleu f1 exact_match \
    --visualize \
    --trend_analysis \
    --format md
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--eval_dirs` | Directories containing evaluation results |
| `--eval_files` | Specific evaluation result files |
| `--output_dir` | Directory to save comparison results (default: `data/comparisons/`) |
| `--metrics` | Metrics to include in comparison (default: perplexity, rouge, bleu, f1, exact_match) |
| `--visualize` | Generate visualizations of comparisons |
| `--format` | Output format for the report: md, html, csv, json (default: md) |
| `--trend_analysis` | Show performance trends across model versions |

## Examples

### Basic Usage

Compare all evaluation results in the default directory:

```bash
python scripts/compare_models.py
```

### Specific Evaluation Files

Compare specific evaluation files:

```bash
python scripts/compare_models.py \
    --eval_files data/evaluation/model1_eval.json data/evaluation/model2_eval.json
```

### Generate Visualizations

Compare models and generate visualizations:

```bash
python scripts/compare_models.py --visualize
```

### Trend Analysis

Track performance trends across model versions:

```bash
python scripts/compare_models.py --trend_analysis
```

### HTML Report

Generate an HTML report for easier viewing:

```bash
python scripts/compare_models.py --format html
```

### Specific Metrics

Focus on specific metrics for comparison:

```bash
python scripts/compare_models.py --metrics perplexity rouge_l_f1 bleu
```

## Output

The script produces the following outputs in the specified output directory:

### Reports

Reports are saved in the `reports/` subdirectory:

- `model_comparison_<timestamp>.<format>`: Comprehensive comparison report

### Visualizations

If `--visualize` is enabled, visualizations are saved in the `visualizations/` subdirectory:

- `metric_<metric_name>.png`: Bar charts for each metric
- `radar_chart.png`: Radar chart comparing top models
- `model_ranking.png`: Overall model ranking chart

### Trend Analysis

If `--trend_analysis` is enabled, trend analysis is saved in the `trends/` subdirectory:

- `trend_<metric_name>.png`: Line charts showing metric trends over time
- `trend_analysis.md`: Markdown report of trend analysis

## Report Format

The report includes:

1. **Summary**
   - Number of models compared
   - Top performing model

2. **Rankings**
   - Table of models with metrics and ranks
   - Highlighting of best results for each metric

3. **Visualizations**
   - References to the visualization files

## Example Report (Markdown)

```markdown
# Model Comparison Report

Generated: 2025-03-22 08:30:00

## Summary

**Number of Models Compared:** 3

**Top Performing Model:** llama-3-ft-model-v2

## Rankings

| Rank | Model | Perplexity | ROUGE-L F1 | BLEU | F1 | Avg Rank |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | llama-3-ft-model-v2 | **12.3456** | 0.7823 | **0.6542** | 0.8123 | **1.50** |
| 2 | llama-3-ft-model-v1 | 15.6789 | **0.8012** | 0.6234 | 0.7845 | 2.25 |
| 3 | llama-3-base | 32.1098 | 0.6543 | 0.5432 | **0.8245** | 2.50 |

_Note: For perplexity metrics, lower values are better. For all other metrics, higher values are better._

## Visualizations

Visual comparisons are available in the 'visualizations' directory.
```

## Integration with Other Scripts

This utility is typically used after running model evaluations:

1. Fine-tune models: `run_finetuning.py`
2. Evaluate models: `evaluate_model.py` or `advanced_evaluate.py`
3. Compare models: `compare_models.py`

Example workflow:

```bash
# Run evaluation on model v1
python scripts/evaluate_model.py \
    --model_path data/models/llama-3-ft-v1/ \
    --test_data data/processed/dataset/test.jsonl \
    --output_dir data/evaluation/

# Run evaluation on model v2
python scripts/evaluate_model.py \
    --model_path data/models/llama-3-ft-v2/ \
    --test_data data/processed/dataset/test.jsonl \
    --output_dir data/evaluation/

# Compare the models
python scripts/compare_models.py --visualize --trend_analysis
```

## Tips for Effective Use

1. **Consistent Evaluation**: Ensure all models are evaluated on the same dataset and with the same metrics for fair comparison.

2. **Timestamps**: Add timestamps to model evaluations to enable trend analysis functionality.

3. **Multiple Metrics**: Consider multiple metrics rather than focusing on a single metric to get a more comprehensive view of model performance.

4. **Visualization**: Use the `--visualize` flag to generate visual comparisons, which can make it easier to spot patterns and communicate results.

5. **Regular Tracking**: Run comparisons regularly during the fine-tuning process to track progress and identify performance plateaus.
