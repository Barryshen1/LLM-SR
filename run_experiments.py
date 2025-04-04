import os
import argparse
from datetime import datetime
import json
from run_experiment import run_experiment
from evaluate_results import load_samples, evaluate_results

def run_experiments(use_api=False, api_model="gpt-3.5-turbo", n_samples_per_problem=25, n_worst=10):
    """
    Run LLM-SR experiments on multiple problems and identify the worst-performing samples.
    
    Args:
        use_api: Whether to use the OpenAI API
        api_model: The API model to use if use_api is True
        n_samples_per_problem: Target number of samples to generate per problem
        n_worst: Number of worst samples to identify
    """
    # Create experiments directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_dir = f"./experiments_{timestamp}"
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Define problems and specifications
    problems = {
        "oscillator1": "./specs/specification_oscillator1_numpy.txt",
        "oscillator2": "./specs/specification_oscillator2_numpy.txt",
        "bactgrow": "./specs/specification_bactgrow_numpy.txt",
        "stressstrain": "./specs/specification_stressstrain_numpy.txt"
    }
    
    all_samples = []
    problem_log_dirs = {}
    
    # Run experiments for each problem
    for problem_name, spec_path in problems.items():
        log_dir = os.path.join(experiments_dir, problem_name)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\nRunning experiment for {problem_name} with target {n_samples_per_problem} samples")
        
        try:
            # Run the experiment
            run_experiment(
                problem_name=problem_name,
                spec_path=spec_path,
                log_path=log_dir,
                use_api=use_api,
                api_model=api_model,
                max_sample_nums=n_samples_per_problem
            )
            
            print(f"Experiment for {problem_name} completed")
            
            # Store the log directory
            problem_log_dirs[problem_name] = log_dir
            
            # Load samples from this experiment
            samples = load_samples(log_dir)
            for sample in samples:
                sample['problem_name'] = problem_name
            
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"Error running experiment for {problem_name}: {e}")
    
    # Process all samples to find the worst ones
    print(f"\nLoaded {len(all_samples)} total samples from all experiments")
    
    # Filter samples with valid scores
    valid_samples = [s for s in all_samples if s.get('score') is not None]
    
    print(f"Found {len(valid_samples)} valid samples with scores")
    
    # Sort by score (ascending - lower is worse)
    valid_samples.sort(key=lambda x: x['score'])
    
    # Take the n_worst worst samples
    worst_samples = valid_samples[:n_worst]
    
    # Print the worst samples
    print(f"\n===== {n_worst} WORST-PERFORMING SAMPLES ACROSS ALL PROBLEMS =====")
    for i, sample in enumerate(worst_samples):
        print(f"\nWorst Sample #{i+1}:")
        print(f"Problem: {sample['problem_name']}")
        print(f"Log Directory: {sample['log_dir']}")
        print(f"Filename: {sample['filename']}")
        print(f"Sample Order: {sample.get('sample_order', 'N/A')}")
        print(f"Score: {sample['score']}")
        print("Function:")
        print("-" * 50)
        print(sample['function'])
        print("-" * 50)
    
    # Save results to a file
    results_file = os.path.join(experiments_dir, "worst_samples_overall.json")
    with open(results_file, 'w') as f:
        json.dump(worst_samples, f, indent=2)
    
    print(f"\nSaved worst samples to {results_file}")
    
    # Generate a summary file with information about why these samples performed poorly
    analysis_file = os.path.join(experiments_dir, "analysis_of_poor_performance.txt")
    with open(analysis_file, 'w') as f:
        f.write("ANALYSIS OF POORLY PERFORMING SAMPLES\n")
        f.write("====================================\n\n")
        f.write(f"Total samples analyzed: {len(all_samples)}\n")
        f.write(f"Valid samples with scores: {len(valid_samples)}\n\n")
        
        f.write("Problem-specific statistics:\n")
        for problem_name in problems:
            problem_samples = [s for s in valid_samples if s['problem_name'] == problem_name]
            if problem_samples:
                avg_score = sum(s['score'] for s in problem_samples) / len(problem_samples)
                f.write(f"  {problem_name}: {len(problem_samples)} samples, avg score: {avg_score:.6f}\n")
        
        f.write("\nWorst Samples Analysis:\n")
        for i, sample in enumerate(worst_samples):
            f.write(f"\n{i+1}. Problem: {sample['problem_name']}, Score: {sample['score']:.6f}\n")
            f.write(f"   Function: {sample['function'].split('return')[1].strip() if 'return' in sample['function'] else 'Complex function'}\n")
            
            # Add potential reasons for poor performance
            if 'function' in sample:
                func_text = sample['function'].lower()
                if '**' in func_text or '^' in func_text:
                    f.write("   Potential issue: Incorrect power operation syntax\n")
                if 'nan' in func_text or 'inf' in func_text:
                    f.write("   Potential issue: Possible numerical instability\n")
                if 'if ' in func_text or 'else' in func_text:
                    f.write("   Potential issue: Conditional logic may cause fitting problems\n")
                if func_text.count('return') > 1:
                    f.write("   Potential issue: Multiple return statements\n")
    
    print(f"Analysis of poor performance saved to {analysis_file}")
    
    # Also evaluate each problem individually
    for problem_name, log_dir in problem_log_dirs.items():
        print(f"\nEvaluating results for {problem_name}...")
        evaluate_results(log_dir, n_worst, visualize=True)
    
    return worst_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LLM-SR experiments and identify poorly performing samples')
    parser.add_argument('--use_api', action='store_true', help='Use the OpenAI API')
    parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo", help='API model to use')
    parser.add_argument('--n_samples_per_problem', type=int, default=25, help='Target number of samples per problem')
    parser.add_argument('--n_worst', type=int, default=10, help='Number of worst samples to identify')
    
    args = parser.parse_args()
    
    run_experiments(
        use_api=args.use_api,
        api_model=args.api_model,
        n_samples_per_problem=args.n_samples_per_problem,
        n_worst=args.n_worst
    )
