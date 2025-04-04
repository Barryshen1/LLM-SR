import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime

def load_results(experiment_dir):
    """Load all experiment results from the given directory."""
    # Load the summary file
    with open(os.path.join(experiment_dir, 'summary.json'), 'r') as f:
        summary = json.load(f)
    
    # Load all samples
    with open(os.path.join(experiment_dir, 'all_samples.json'), 'r') as f:
        all_samples = json.load(f)
    
    return summary, all_samples

def compute_statistics(samples):
    """Compute statistics from the experiment samples."""
    # Extract scores, filtering out None values
    scores = [s['score'] for s in samples if s['score'] is not None]
    
    if not scores:
        return {
            'num_valid_samples': 0,
            'best_score': None,
            'median_score': None,
            'mean_score': None,
            'worst_score': None,
            'std_dev': None
        }
    
    stats = {
        'num_valid_samples': len(scores),
        'best_score': max(scores),
        'median_score': np.median(scores),
        'mean_score': np.mean(scores),
        'worst_score': min(scores),
        'std_dev': np.std(scores)
    }
    
    return stats

def plot_score_progression(samples, output_dir):
    """Plot the progression of scores over samples."""
    # Extract sample numbers and scores
    sample_nums = [s['sample_num'] for s in samples if s['score'] is not None]
    scores = [s['score'] for s in samples if s['score'] is not None]
    
    if not scores:
        print("No valid scores to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sample_nums, scores, marker='o', linestyle='-', alpha=0.6)
    
    # Add best score line
    best_score = max(scores)
    plt.axhline(y=best_score, color='r', linestyle='--', alpha=0.7, 
                label=f'Best Score: {best_score:.6f}')
    
    # Format the plot
    plt.title('LLM-SR Score Progression', fontsize=14)
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Score (negative MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis to only show integer sample numbers
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_progression.png'), dpi=300)
    plt.close()

def plot_score_distribution(samples, output_dir):
    """Plot the distribution of scores."""
    # Extract scores
    scores = [s['score'] for s in samples if s['score'] is not None]
    
    if not scores:
        print("No valid scores to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram with KDE
    sns.histplot(scores, kde=True, bins=20)
    
    # Format the plot
    plt.title('Distribution of Scores', fontsize=14)
    plt.xlabel('Score (negative MSE)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300)
    plt.close()

def plot_cumulative_best_score(samples, output_dir):
    """Plot the cumulative best score over samples."""
    # Extract sample numbers and scores
    sample_nums = [s['sample_num'] for s in samples if s['score'] is not None]
    scores = [s['score'] for s in samples if s['score'] is not None]
    
    if not scores:
        print("No valid scores to plot")
        return
    
    # Calculate cumulative best score
    best_scores = []
    current_best = float('-inf')
    
    for score in scores:
        current_best = max(current_best, score)
        best_scores.append(current_best)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sample_nums, best_scores, marker='o', linestyle='-', color='green', alpha=0.8)
    
    # Format the plot
    plt.title('Cumulative Best Score', fontsize=14)
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Best Score So Far', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to only show integer sample numbers
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_best_score.png'), dpi=300)
    plt.close()

def plot_computation_time(samples, output_dir):
    """Plot the computation time over samples."""
    # Extract sample numbers and times
    sample_nums = [s['sample_num'] for s in samples]
    sample_times = [s['sample_time'] for s in samples if s['sample_time'] is not None]
    evaluate_times = [s['evaluate_time'] for s in samples if s['evaluate_time'] is not None]
    
    if not sample_times or not evaluate_times:
        print("No valid time data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    if sample_times:
        plt.plot(sample_nums[:len(sample_times)], sample_times, 
                marker='o', linestyle='-', label='Sampling Time', alpha=0.6)
    
    if evaluate_times:
        plt.plot(sample_nums[:len(evaluate_times)], evaluate_times, 
                marker='x', linestyle='-', label='Evaluation Time', alpha=0.6)
    
    # Format the plot
    plt.title('Computation Time per Sample', fontsize=14)
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis to only show integer sample numbers
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computation_time.png'), dpi=300)
    plt.close()

def save_top_functions(samples, output_dir, top_n=10):
    """Save the top N functions to a file."""
    # Filter out samples with None scores
    valid_samples = [s for s in samples if s['score'] is not None]
    
    # Sort by score (descending)
    sorted_samples = sorted(valid_samples, key=lambda x: x['score'], reverse=True)
    
    # Create a formatted string with the top functions
    top_functions_text = f"# Top {top_n} Functions Generated by LLM-SR\n\n"
    top_functions_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, sample in enumerate(sorted_samples[:top_n]):
        top_functions_text += f"## {i+1}. Sample #{sample['sample_num']} (Score: {sample['score']:.6f})\n\n"
        top_functions_text += "```python\n"
        top_functions_text += sample['function'].strip() + "\n"
        top_functions_text += "```\n\n"
    
    # Save to file
    with open(os.path.join(output_dir, 'top_functions.md'), 'w') as f:
        f.write(top_functions_text)

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM-SR experiment results')
    parser.add_argument('--experiment_dir', type=str, required=True, 
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (defaults to experiment_dir/evaluation)')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'evaluation')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the results
    summary, all_samples = load_results(args.experiment_dir)
    
    # Compute statistics
    stats = compute_statistics(all_samples)
    
    # Create plots
    plot_score_progression(all_samples, args.output_dir)
    plot_score_distribution(all_samples, args.output_dir)
    plot_cumulative_best_score(all_samples, args.output_dir)
    plot_computation_time(all_samples, args.output_dir)
    
    # Save top functions
    save_top_functions(all_samples, args.output_dir)
    
    # Create a summary report
    report = {
        'experiment_summary': summary,
        'statistics': stats,
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save the report
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a human-readable report
    report_text = "# LLM-SR Experiment Evaluation\n\n"
    report_text += f"Evaluated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_text += f"## Experiment Summary\n\n"
    report_text += f"- Total samples: {summary['total_samples']}\n"
    report_text += f"- Elapsed time: {summary['elapsed_time']:.2f} seconds\n"
    report_text += f"- Best score: {summary['best_score']}\n\n"
    
    report_text += f"## Best Function\n\n"
    report_text += "```python\n"
    report_text += summary['best_function'].strip() + "\n"
    report_text += "```\n\n"
    
    report_text += f"## Statistics\n\n"
    report_text += f"- Valid samples: {stats['num_valid_samples']}\n"
    report_text += f"- Best score: {stats['best_score']}\n"
    report_text += f"- Median score: {stats['median_score']}\n"
    report_text += f"- Mean score: {stats['mean_score']}\n"
    report_text += f"- Worst score: {stats['worst_score']}\n"
    report_text += f"- Standard deviation: {stats['std_dev']}\n\n"
    
    report_text += f"## Plots\n\n"
    report_text += f"- [Score Progression](score_progression.png)\n"
    report_text += f"- [Score Distribution](score_distribution.png)\n"
    report_text += f"- [Cumulative Best Score](cumulative_best_score.png)\n"
    report_text += f"- [Computation Time](computation_time.png)\n\n"
    
    report_text += f"See [top_functions.md](top_functions.md) for the top 10 functions generated.\n"
    
    # Save the report
    with open(os.path.join(args.output_dir, 'evaluation_report.md'), 'w') as f:
        f.write(report_text)
    
    print(f"Evaluation complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
