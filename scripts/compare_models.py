#!/usr/bin/env python
"""
Model Performance Comparison Tool for Llama 3 fine-tuned models.

This script allows quick comparison of performance metrics across different model versions 
without running full evaluations, by loading existing evaluation results.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_comparison.log')
    ]
)
logger = logging.getLogger('model_comparison')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare performance of fine-tuned Llama models')
    
    parser.add_argument(
        '--eval_dirs',
        type=str,
        nargs='+',
        help='Directories containing evaluation results'
    )
    
    parser.add_argument(
        '--eval_files',
        type=str,
        nargs='+',
        help='Specific evaluation result files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/comparisons',
        help='Directory to save comparison results'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['perplexity', 'rouge', 'bleu', 'f1', 'exact_match'],
        help='Metrics to include in comparison'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations of comparisons'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='md',
        choices=['md', 'html', 'csv', 'json'],
        help='Output format for the report'
    )
    
    parser.add_argument(
        '--trend_analysis',
        action='store_true',
        help='Show performance trends across model versions'
    )
    
    return parser.parse_args()


def find_evaluation_files(eval_dirs=None, eval_files=None):
    """
    Find evaluation result files from specified directories and files.
    
    Args:
        eval_dirs: List of directories containing evaluation results
        eval_files: List of specific evaluation files
        
    Returns:
        List of paths to evaluation files
    """
    logger.info("Finding evaluation files")
    result_files = []
    
    # Process directories
    if eval_dirs:
        for directory in eval_dirs:
            # Look for JSON files in the directory and its subdirectories
            pattern = os.path.join(directory, '**', '*.json')
            directory_files = glob.glob(pattern, recursive=True)
            result_files.extend(directory_files)
    
    # Add specific files
    if eval_files:
        result_files.extend(eval_files)
    
    # Remove duplicates
    result_files = list(set(result_files))
    logger.info(f"Found {len(result_files)} evaluation files")
    
    return result_files


def load_evaluation_results(file_paths):
    """
    Load evaluation results from files.
    
    Args:
        file_paths: List of paths to evaluation files
        
    Returns:
        Dictionary mapping model names to evaluation results
    """
    logger.info(f"Loading evaluation results from {len(file_paths)} files")
    results = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a direct evaluation result
            if 'model_name' in data and 'metrics' in data:
                model_name = data['model_name']
                results[model_name] = data
                logger.info(f"Loaded evaluation result for {model_name}")
            
            # Check if this is a results directory containing multiple evaluations
            elif 'models' in data:
                for model_name, model_data in data['models'].items():
                    results[model_name] = model_data
                    logger.info(f"Loaded evaluation result for {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    logger.info(f"Loaded results for {len(results)} models")
    return results


def extract_metric_values(results, metrics):
    """
    Extract key metric values from evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        metrics: List of metrics to extract
        
    Returns:
        DataFrame with metric values for each model
    """
    logger.info("Extracting metric values")
    model_metrics = []
    
    for model_name, data in results.items():
        model_data = {'model': model_name}
        
        # Extract timestamp if available
        if 'timestamp' in data:
            model_data['timestamp'] = data['timestamp']
        
        for metric in metrics:
            if metric in data.get('metrics', {}):
                metric_data = data['metrics'][metric]
                
                # Extract appropriate values based on metric type
                if metric == 'perplexity':
                    model_data[f'{metric}'] = metric_data.get('perplexity', float('nan'))
                
                elif metric == 'rouge':
                    # Extract ROUGE-L F1
                    rouge_l = metric_data.get('rouge-l', {})
                    model_data[f'{metric}_l_f1'] = rouge_l.get('fmeasure', float('nan'))
                    
                    # Extract ROUGE-1 and ROUGE-2 for completeness
                    rouge_1 = metric_data.get('rouge-1', {})
                    rouge_2 = metric_data.get('rouge-2', {})
                    model_data[f'{metric}_1_f1'] = rouge_1.get('fmeasure', float('nan'))
                    model_data[f'{metric}_2_f1'] = rouge_2.get('fmeasure', float('nan'))
                
                elif metric == 'bleu':
                    model_data[f'{metric}'] = metric_data.get('bleu', float('nan'))
                
                elif metric in ['exact_match', 'accuracy']:
                    model_data[f'{metric}'] = metric_data
                
                elif metric == 'f1':
                    model_data[f'{metric}'] = metric_data.get('f1', float('nan'))
                    model_data[f'{metric}_precision'] = metric_data.get('precision', float('nan'))
                    model_data[f'{metric}_recall'] = metric_data.get('recall', float('nan'))
                
                else:
                    # For other metrics, try to extract the main value
                    if isinstance(metric_data, dict):
                        # Use the first value in the dictionary
                        first_key = next(iter(metric_data))
                        model_data[f'{metric}'] = metric_data[first_key]
                    else:
                        model_data[f'{metric}'] = metric_data
        
        model_metrics.append(model_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(model_metrics)
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    logger.info(f"Extracted metrics for {len(df)} models")
    return df


def rank_models(df):
    """
    Rank models based on each metric.
    
    Args:
        df: DataFrame with metric values
        
    Returns:
        DataFrame with added rank columns
    """
    logger.info("Ranking models by metrics")
    
    # Create a copy to avoid modifying the original
    ranked_df = df.copy()
    
    # Find metric columns (exclude 'model' and 'timestamp')
    metric_cols = [col for col in df.columns if col not in ['model', 'timestamp']]
    
    # Add rank for each metric
    for col in metric_cols:
        # For perplexity, lower is better; for others, higher is better
        ascending = 'perplexity' in col
        ranked_df[f'{col}_rank'] = ranked_df[col].rank(ascending=ascending)
    
    # Calculate average rank
    rank_cols = [f'{col}_rank' for col in metric_cols]
    if rank_cols:
        ranked_df['avg_rank'] = ranked_df[rank_cols].mean(axis=1)
        
        # Sort by average rank
        ranked_df = ranked_df.sort_values('avg_rank')
    
    return ranked_df


def visualize_comparison(df, ranked_df, output_dir):
    """
    Create visualizations of model comparisons.
    
    Args:
        df: DataFrame with metric values
        ranked_df: DataFrame with added ranks
        output_dir: Directory to save visualizations
    """
    logger.info("Generating visualizations")
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Use a consistent style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Find metric columns (exclude 'model', 'timestamp', and rank columns)
    metric_cols = [col for col in df.columns 
                   if col not in ['model', 'timestamp']
                   and not col.endswith('_rank')]
    
    # 1. Bar charts for each metric
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        
        # Sort by metric value
        # For perplexity, lower is better; for others, higher is better
        ascending = 'perplexity' in metric
        sorted_df = df.sort_values(metric, ascending=ascending)
        
        # Select top 10 models for readability
        plot_df = sorted_df.head(10)
        
        # Create bar chart
        ax = sns.barplot(x=metric, y='model', data=plot_df)
        
        # Set title and labels
        title = f'Model Comparison - {metric.replace("_", " ").title()}'
        if 'perplexity' in metric:
            title += ' (lower is better)'
        else:
            title += ' (higher is better)'
        
        plt.title(title)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, f'metric_{metric}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Radar chart of top models
    if len(ranked_df) > 1 and len(metric_cols) > 2:
        try:
            # Select top 5 models by average rank
            top_models = ranked_df.head(5)['model'].tolist()
            
            # Prepare data for radar chart
            radar_df = df[df['model'].isin(top_models)]
            
            # Create radar chart
            plt.figure(figsize=(10, 10))
            
            # Number of metrics
            categories = metric_cols
            N = len(categories)
            
            # Create angle for each metric
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize the plot
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per metric and add labels
            plt.xticks(angles[:-1], categories, size=8)
            
            # Draw the chart for each model
            for i, model in enumerate(top_models):
                model_data = radar_df[radar_df['model'] == model]
                
                # Get values and scale them to 0-1 range for radar chart
                values = []
                for metric in categories:
                    value = model_data[metric].values[0]
                    
                    # For perplexity, lower is better so invert the scale
                    if 'perplexity' in metric:
                        # Invert and normalize to 0-1 range
                        max_val = df[metric].max()
                        min_val = df[metric].min()
                        value = 1 - ((value - min_val) / (max_val - min_val) if max_val > min_val else 0)
                    else:
                        # Normalize to 0-1 range
                        max_val = df[metric].max()
                        min_val = df[metric].min()
                        value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
                    
                    values.append(value)
                
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Top Models Comparison (Normalized Metrics)')
            
            # Save the radar chart
            plt.savefig(os.path.join(viz_dir, 'radar_chart.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")
    
    # 3. Overall ranking chart
    plt.figure(figsize=(10, 6))
    
    # Select top 10 models by average rank
    top_ranked = ranked_df.head(10)
    
    # Sort by average rank
    top_ranked = top_ranked.sort_values('avg_rank')
    
    # Create bar chart
    ax = sns.barplot(x='avg_rank', y='model', data=top_ranked)
    
    plt.title('Model Ranking (lower is better)')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(viz_dir, 'model_ranking.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {viz_dir}")


def analyze_trends(df, output_dir):
    """
    Analyze performance trends across model versions.
    
    Args:
        df: DataFrame with metric values and timestamps
        output_dir: Directory to save trend analysis
    """
    if 'timestamp' not in df.columns or len(df) < 2:
        logger.warning("Trend analysis requires timestamp information and multiple models")
        return
    
    logger.info("Analyzing performance trends")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Find metric columns
    metric_cols = [col for col in df.columns 
                   if col not in ['model', 'timestamp']
                   and not col.endswith('_rank')]
    
    # Create visualizations directory
    trend_dir = os.path.join(output_dir, 'trends')
    os.makedirs(trend_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Line charts for metric trends
    for metric in metric_cols:
        plt.figure(figsize=(12, 6))
        
        # Plot trend line
        plt.plot(df['timestamp'], df[metric], 'o-', linewidth=2, markersize=8)
        
        # Add model names as annotations
        for i, row in df.iterrows():
            plt.annotate(
                row['model'],
                (row['timestamp'], row[metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        # Set title and labels
        title = f'Trend Analysis - {metric.replace("_", " ").title()}'
        if 'perplexity' in metric:
            title += ' (lower is better)'
        else:
            title += ' (higher is better)'
            
        plt.title(title)
        plt.xlabel('Timestamp')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(trend_dir, f'trend_{metric}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Generate a trend report
    trend_report_path = os.path.join(trend_dir, 'trend_analysis.md')
    with open(trend_report_path, 'w') as f:
        f.write("# Model Performance Trend Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall trends summary
        f.write("## Summary of Performance Trends\n\n")
        
        # Calculate improvement percentage for each metric
        if len(df) >= 2:
            first_model = df.iloc[0]
            latest_model = df.iloc[-1]
            
            f.write(f"Comparing first model ({first_model['model']}) to latest model ({latest_model['model']}):\n\n")
            f.write("| Metric | First Model | Latest Model | Change | % Change |\n")
            f.write("|--------|-------------|--------------|--------|---------|\n")
            
            for metric in metric_cols:
                first_val = first_model[metric]
                latest_val = latest_model[metric]
                
                absolute_change = latest_val - first_val
                
                # Calculate percentage change
                if first_val != 0:
                    percentage_change = (absolute_change / abs(first_val)) * 100
                else:
                    percentage_change = float('inf') if absolute_change > 0 else float('-inf') if absolute_change < 0 else 0
                
                # Determine if this is an improvement
                is_improvement = False
                if 'perplexity' in metric:  # Lower is better
                    is_improvement = absolute_change < 0
                else:  # Higher is better
                    is_improvement = absolute_change > 0
                
                # Format change strings
                change_str = f"{absolute_change:.4f}"
                percentage_str = f"{percentage_change:.2f}%"
                
                # Add improvement indicator
                if is_improvement:
                    change_str = f"{change_str} ✓"
                    percentage_str = f"{percentage_str} ✓"
                
                f.write(f"| {metric} | {first_val:.4f} | {latest_val:.4f} | {change_str} | {percentage_str} |\n")
        
        # Add trend visualizations reference
        f.write("\n\nDetailed trend visualizations are available in the 'trends' directory.\n")
    
    logger.info(f"Trend analysis saved to {trend_dir}")


def generate_report(df, ranked_df, output_dir, format='md'):
    """
    Generate a comparison report.
    
    Args:
        df: DataFrame with metric values
        ranked_df: DataFrame with added ranks
        output_dir: Directory to save the report
        format: Output format (md, html, csv, json)
        
    Returns:
        Path to the generated report
    """
    logger.info(f"Generating {format} comparison report")
    
    report_dir = os.path.join(output_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"model_comparison_{timestamp}.{format}")
    
    # Find metric columns
    metric_cols = [col for col in df.columns 
                  if col not in ['model', 'timestamp']
                  and not col.endswith('_rank')]
    
    # Get top model
    top_model = ranked_df.iloc[0]['model'] if not ranked_df.empty else "N/A"
    
    if format == 'json':
        # JSON report
        report_data = {
            'timestamp': timestamp,
            'models': df.to_dict(orient='records'),
            'rankings': ranked_df.to_dict(orient='records'),
            'top_model': top_model
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    elif format == 'csv':
        # Save to CSV
        ranked_df.to_csv(report_path, index=False)
    
    elif format == 'html':
        # HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; margin: 15px 0; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ font-weight: bold; color: #2a9d8f; }}
                .metric-note {{ color: #666; font-style: italic; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <p><strong>Number of Models Compared:</strong> {len(df)}</p>
            <p><strong>Top Performing Model:</strong> <span class="best">{top_model}</span></p>
            
            <h2>Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
        """
        
        # Add metric columns to header
        for metric in metric_cols:
            html_content += f'<th>{metric.replace("_", " ").title()}</th>\n'
        
        html_content += '<th>Avg Rank</th></tr>\n'
        
        # Add rows for each model
        for i, (_, row) in enumerate(ranked_df.iterrows()):
            html_content += f'<tr><td>{i+1}</td><td>{row["model"]}</td>\n'
            
            for metric in metric_cols:
                # Highlight the best value for each metric
                is_best = row[f"{metric}_rank"] == 1
                
                value = row[metric]
                value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                if is_best:
                    html_content += f'<td class="best">{value_str}</td>\n'
                else:
                    html_content += f'<td>{value_str}</td>\n'
            
            # Add average rank
            avg_rank = row['avg_rank']
            class_attr = ' class="best"' if i == 0 else ''
            html_content += f'<td{class_attr}>{avg_rank:.2f}</td></tr>\n'
        
        html_content += """
            </table>
            
            <p class="metric-note">Note: For perplexity metrics, lower values are better. For all other metrics, higher values are better.</p>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    else:  # Markdown format
        with open(report_path, 'w') as f:
            f.write(f"# Model Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"**Number of Models Compared:** {len(df)}\n\n")
            f.write(f"**Top Performing Model:** {top_model}\n\n")
            
            f.write("## Rankings\n\n")
            
            # Create table header
            f.write("| Rank | Model |")
            for metric in metric_cols:
                f.write(f" {metric.replace('_', ' ').title()} |")
            f.write(" Avg Rank |\n")
            
            # Table separator
            f.write("| --- | --- |")
            for _ in metric_cols:
                f.write(" --- |")
            f.write(" --- |\n")
            
            # Add rows for each model
            for i, (_, row) in enumerate(ranked_df.iterrows()):
                f.write(f"| {i+1} | {row['model']} |")
                
                for metric in metric_cols:
                    # Format the value
                    value = row[metric]
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    
                    # Highlight the best value for each metric
                    is_best = row[f"{metric}_rank"] == 1
                    if is_best:
                        value_str = f"**{value_str}**"
                    
                    f.write(f" {value_str} |")
                
                # Add average rank
                avg_rank = row['avg_rank']
                rank_str = f"**{avg_rank:.2f}**" if i == 0 else f"{avg_rank:.2f}"
                f.write(f" {rank_str} |\n")
            
            f.write("\n")
            f.write("_Note: For perplexity metrics, lower values are better. For all other metrics, higher values are better._\n\n")
            
            # Add references to visualizations if created
            viz_dir = os.path.join(output_dir, 'visualizations')
            if os.path.exists(viz_dir):
                f.write("## Visualizations\n\n")
                f.write("Visual comparisons are available in the 'visualizations' directory.\n")
    
    logger.info(f"Comparison report generated at {report_path}")
    return report_path


def main():
    """Main entry point for model comparison."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Find evaluation files
        if not args.eval_dirs and not args.eval_files:
            # Default to looking in standard evaluation directories
            default_dirs = ['data/evaluation', 'data/evaluation/advanced', 'outputs/evaluation']
            eval_files = find_evaluation_files(default_dirs)
        else:
            eval_files = find_evaluation_files(args.eval_dirs, args.eval_files)
        
        if not eval_files:
            logger.error("No evaluation files found. Please specify directories or files.")
            sys.exit(1)
        
        # Load evaluation results
        evaluation_results = load_evaluation_results(eval_files)
        
        if not evaluation_results:
            logger.error("No valid evaluation results found in the specified files.")
            sys.exit(1)
        
        # Extract metric values
        metrics_df = extract_metric_values(evaluation_results, args.metrics)
        
        # Rank models
        ranked_df = rank_models(metrics_df)
        
        # Generate report
        report_path = generate_report(metrics_df, ranked_df, args.output_dir, args.format)
        
        # Create visualizations if requested
        if args.visualize:
            visualize_comparison(metrics_df, ranked_df, args.output_dir)
        
        # Perform trend analysis if requested
        if args.trend_analysis and 'timestamp' in metrics_df.columns:
            analyze_trends(metrics_df, args.output_dir)
        
        logger.info(f"Model comparison completed successfully. Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
