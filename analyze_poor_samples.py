#!/usr/bin/env python3
"""
Script to analyze and print poorly performing samples from LLM-SR experiments.
This helps identify patterns in ineffective equation skeletons and understand
why certain mathematical forms may not work well for specific problems.
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt


def load_samples(log_dir: str) -> List[Dict[str, Any]]:
    """
    Load all samples from the specified log directory.
    
    Args:
        log_dir: Path to the log directory containing sample JSON files
        
    Returns:
        List of sample dictionaries
    """
    samples_dir = os.path.join(log_dir, 'samples')
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    samples = []
    for filename in os.listdir(samples_dir):
        if filename.startswith('samples_') and filename.endswith('.json'):
            file_path = os.path.join(samples_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                    # Some samples might not have a score if they failed to run
                    if sample.get('score') is not None:
                        samples.append(sample)
            except json.JSONDecodeError:
                print(f"Error parsing JSON from {file_path}")
                
    return samples


def extract_function_body(function_str: str) -> str:
    """
    Extract just the body of the function without the function definition.
    
    Args:
        function_str: String containing the full function
        
    Returns:
        The function body
    """
    lines = function_str.strip().split('\n')
    # Skip the function definition and any docstring
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            body_start = i + 1
            break
    
    # Skip docstring if present
    if body_start < len(lines) and '"""' in lines[body_start]:
        for i in range(body_start, len(lines)):
            if i > body_start and '"""' in lines[i]:
                body_start = i + 1
                break
    
    return '\n'.join(lines[body_start:])


def analyze_equation_pattern(function_body: str) -> Dict[str, Any]:
    """
    Analyze the equation pattern used in the function body.
    
    Args:
        function_body: String containing the function body
        
    Returns:
        Dictionary of analysis results
    """
    analysis = {
        'num_lines': len(function_body.strip().split('\n')),
        'uses_conditionals': 'if ' in function_body,
        'uses_loops': any(loop in function_body for loop in ['for ', 'while ']),
        'uses_nonlinear': any(term in function_body for term in 
                             ['np.sin', 'np.cos', 'np.exp', '**', 'np.log', 'np.sqrt']),
        'complexity': 0,  # Will be calculated below
        'params_used': []
    }
    
    # Extract which parameters are used
    param_pattern = re.compile(r'params\[(\d+)\]')
    params_used = param_pattern.findall(function_body)
    analysis['params_used'] = sorted([int(p) for p in params_used])
    analysis['num_params'] = len(analysis['params_used'])
    
    # Calculate rough complexity score based on operators and function calls
    operators = ['+', '-', '*', '/', '**', 'np.']
    for op in operators:
        analysis['complexity'] += function_body.count(op)
    
    return analysis


def identify_potential_issues(sample: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
    """
    Identify potential issues that might cause poor performance.
    
    Args:
        sample: Dictionary containing sample data
        analysis: Dictionary of analysis results
        
    Returns:
        List of potential issues
    """
    issues = []
    
    # Low score may indicate a fundamental mismatch with the problem
    if sample.get('score', 0) is None or sample.get('score', 0) < -10:
        issues.append("Very low score indicates poor fit to data")
    
    # Complex equations can overfit or be challenging to optimize
    if analysis['complexity'] > 15:
        issues.append("High complexity may lead to optimization difficulty")
    
    # Not using enough parameters
    if analysis['num_params'] < 3:
        issues.append("Using too few parameters may limit expressivity")
    
    # Using too many parameters can lead to overfitting
    if analysis['num_params'] > 8:
        issues.append("Using many parameters may cause overfitting")
    
    # Linear models might be too simple for complex relationships
    if not analysis['uses_nonlinear'] and not analysis['uses_conditionals']:
        issues.append("Linear model may be too simple for nonlinear phenomena")
    
    # Complex control flow can make optimization harder
    if analysis['uses_conditionals'] or analysis['uses_loops']:
        issues.append("Control flow structures may cause optimization difficulties")
    
    return issues


def print_poor_samples(samples: List[Dict[str, Any]], 
                      n: int = 10, 
                      score_threshold: Optional[float] = None,
                      verbose: bool = False) -> None:
    """
    Print information about the n worst-performing samples.
    
    Args:
        samples: List of sample dictionaries
        n: Number of samples to print
        score_threshold: Only print samples with score below this threshold
        verbose: Whether to print detailed information
    """
    # Filter out samples without scores (they failed to run)
    valid_samples = [s for s in samples if s.get('score') is not None]
    
    # Sort samples by score (ascending, so worst first)
    sorted_samples = sorted(valid_samples, key=lambda x: x.get('score', float('-inf')))
    
    # Apply score threshold if specified
    if score_threshold is not None:
        sorted_samples = [s for s in sorted_samples if s.get('score', float('-inf')) < score_threshold]
    
    # Limit to n samples
    worst_samples = sorted_samples[:n]
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS OF {len(worst_samples)} WORST PERFORMING SAMPLES")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(worst_samples):
        print(f"\n{'-'*80}")
        print(f"POOR SAMPLE #{i+1} (Order: {sample['sample_order']}, Score: {sample['score']})")
        print(f"{'-'*80}")
        
        function_body = extract_function_body(sample['function'])
        analysis = analyze_equation_pattern(function_body)
        issues = identify_potential_issues(sample, analysis)
        
        print(f"Function body:")
        print(f"{function_body.strip()}")
        
        print(f"\nAnalysis:")
        print(f"- Complexity score: {analysis['complexity']}")
        print(f"- Parameters used: {analysis['params_used']} (total: {analysis['num_params']})")
        print(f"- Uses nonlinear terms: {analysis['uses_nonlinear']}")
        print(f"- Uses conditionals: {analysis['uses_conditionals']}")
        print(f"- Uses loops: {analysis['uses_loops']}")
        
        if issues:
            print(f"\nPotential issues:")
            for issue in issues:
                print(f"- {issue}")
        
        # If verbose, print the full function
        if verbose:
            print(f"\nFull function:")
            print(f"{sample['function']}")
    
    print(f"\n{'='*80}")
    print(f"END OF ANALYSIS")
    print(f"{'='*80}\n")


def analyze_trends(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze trends across all samples to identify patterns.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Dictionary of trend analysis results
    """
    trends = {
        'num_samples': len(samples),
        'complexity_vs_score': [],
        'param_count_vs_score': [],
        'nonlinear_scores': [],
        'linear_scores': [],
        'conditional_scores': [],
        'pattern_counts': defaultdict(int)
    }
    
    for sample in samples:
        if sample.get('score') is None:
            continue
            
        score = sample.get('score', 0)
        function_body = extract_function_body(sample['function'])
        analysis = analyze_equation_pattern(function_body)
        
        trends['complexity_vs_score'].append((analysis['complexity'], score))
        trends['param_count_vs_score'].append((analysis['num_params'], score))
        
        if analysis['uses_nonlinear']:
            trends['nonlinear_scores'].append(score)
        else:
            trends['linear_scores'].append(score)
            
        if analysis['uses_conditionals']:
            trends['conditional_scores'].append(score)
            
        # Identify common patterns based on terms used
        terms = re.findall(r'[a-zA-Z_]+\s*[\+\-\*/]', function_body)
        for term in terms:
            trends['pattern_counts'][term.strip()] += 1
    
    return trends


def plot_trends(trends: Dict[str, Any], output_dir: str) -> None:
    """
    Plot trend analysis results.
    
    Args:
        trends: Dictionary of trend analysis results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot complexity vs score
    plt.figure(figsize=(10, 6))
    if trends['complexity_vs_score']:
        x, y = zip(*trends['complexity_vs_score'])
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('Complexity Score')
        plt.ylabel('Performance Score')
        plt.title('Equation Complexity vs Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'complexity_vs_score.png'))
    
    # Plot parameter count vs score
    plt.figure(figsize=(10, 6))
    if trends['param_count_vs_score']:
        x, y = zip(*trends['param_count_vs_score'])
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('Number of Parameters Used')
        plt.ylabel('Performance Score')
        plt.title('Parameter Count vs Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'param_count_vs_score.png'))
    
    # Plot score distributions for different equation types
    plt.figure(figsize=(12, 6))
    data = [
        trends['linear_scores'],
        trends['nonlinear_scores'], 
        trends['conditional_scores']
    ]
    labels = ['Linear', 'Nonlinear', 'Conditional']
    plt.boxplot(data, labels=labels)
    plt.ylabel('Performance Score')
    plt.title('Performance by Equation Type')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'equation_type_performance.png'))
    
    # Plot most common equation terms
    plt.figure(figsize=(12, 8))
    if trends['pattern_counts']:
        sorted_patterns = sorted(
            trends['pattern_counts'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:15]  # Top 15 patterns
        
        patterns, counts = zip(*sorted_patterns)
        plt.barh(patterns, counts)
        plt.xlabel('Frequency')
        plt.ylabel('Equation Terms')
        plt.title('Most Common Equation Terms')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'common_equation_terms.png'))


def main():
    parser = argparse.ArgumentParser(description='Analyze poor samples from LLM-SR runs')
    parser.add_argument('--log_dir', type=str, required=True, 
                        help='Directory containing the experiment logs')
    parser.add_argument('--n', type=int, default=10, 
                        help='Number of worst samples to print')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Score threshold; only print samples below this score')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print detailed information for each sample')
    parser.add_argument('--analyze_trends', action='store_true', 
                        help='Analyze trends across samples and create plots')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save analysis results and plots')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.log_dir, 'analysis')
    
    try:
        # Load samples
        print(f"Loading samples from {args.log_dir}...")
        samples = load_samples(args.log_dir)
        print(f"Loaded {len(samples)} samples with valid scores.")
        
        # Print poor samples
        print_poor_samples(samples, n=args.n, 
                          score_threshold=args.threshold, 
                          verbose=args.verbose)
        
        # Analyze trends if requested
        if args.analyze_trends:
            print(f"Analyzing trends across all samples...")
            trends = analyze_trends(samples)
            plot_trends(trends, args.output_dir)
            print(f"Analysis complete. Plots saved to {args.output_dir}.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
