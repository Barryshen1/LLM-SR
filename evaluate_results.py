import os
import json
import glob
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def load_samples(log_dir: str) -> List[Dict[str, Any]]:
    """
    Load all samples from a log directory.
    
    Args:
        log_dir: Path to the log directory
        
    Returns:
        List of samples
    """
    samples_dir = os.path.join(log_dir, 'samples')
    
    if not os.path.exists(samples_dir):
        print(f"No samples directory found at {samples_dir}")
        return []
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(samples_dir, '*.json'))
    
    # Load all samples
    all_samples = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                sample = json.load(f)
                # Add log directory and filename for reference
                sample['log_dir'] = log_dir
                sample['filename'] = os.path.basename(file_path)
                all_samples.append(sample)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_samples

def evaluate_results(log_dir: str, n_worst: int = 10, visualize: bool = False) -> List[Dict[str, Any]]:
    """
    Evaluate results from an experiment and identify the worst-performing samples.
    
    Args:
        log_dir: Path to the log directory
        n_worst: Number of worst samples to identify
        visualize: Whether to create visualization of score distribution
        
    Returns:
        List of worst-performing samples
    """
    # Load all samples
    all_samples = load_samples(log_dir)
    
    print(f"Loaded {len(all_samples)} total samples from {log_dir}")
    
    # Filter samples with valid scores (not None)
    valid_samples = [s for s in all_samples if s.get('score') is not None]
    
    print(f"Found {len(valid_samples)} valid samples with scores")
    
    # Sort by score (ascending - lower is worse)
    valid_samples.sort(key=lambda x: x['score'])
    
    # Take the n_worst worst samples
    worst_samples = valid_samples[:n_worst]
    
    # Visualize score distribution if requested
    if visualize and valid_samples:
        scores = [s['score'] for s in valid_samples]
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7)
        plt.axvline(x=worst_samples[-1]['score'], color='r', linestyle='--', 
                   label=f'Worst {n_worst} threshold')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sample Scores')
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(log_dir, 'score_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Score distribution plot saved to {plot_path}")
    
    # Print the worst samples
    print(f"\n===== {len(worst_samples)} WORST-PERFORMING SAMPLES =====")
    for i, sample in enumerate(worst_samples):
        print(f"\nWorst Sample #{i+1}:")
        print(f"Filename: {sample['filename']}")
        print(f"Sample Order: {sample.get('sample_order', 'N/A')}")
        print(f"Score: {sample['score']}")
        print("Function:")
        print("-" * 50)
        print(sample['function'])
        print("-" * 50)
    
    # Save worst samples to a JSON file
    worst_samples_path = os.path.join(log_dir, 'worst_samples.json')
    with open(worst_samples_path, 'w') as f:
        json.dump(worst_samples, f, indent=2)
    print(f"Worst samples saved to {worst_samples_path}")
    
    return worst_samples

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate experiment results')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory')
    parser.add_argument('--n_worst', type=int, default=10, help='Number of worst samples to identify')
    parser.add_argument('--visualize', action='store_true', help='Create visualization of score distribution')
    
    args = parser.parse_args()
    
    evaluate_results(args.log_dir, args.n_worst, args.visualize)
