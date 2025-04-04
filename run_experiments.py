import os
import argparse
import subprocess
import json
import time
from datetime import datetime

def run_experiment(problem_name, spec_path, log_path, num_samples=100, use_api=False, api_model=None):
    """Run a single LLM-SR experiment using run_experiment.py."""
    
    # Construct command
    cmd = ["python", "run_experiment.py", 
           "--problem_name", problem_name,
           "--spec_path", spec_path,
           "--log_path", log_path,
           "--num_samples", str(num_samples)]
    
    if use_api:
        cmd.append("--use_api")
        if api_model:
            cmd.extend(["--api_model", api_model])
    
    # Execute the command
    start_time = time.time()
    print(f"\n{'-'*80}")
    print(f"Running experiment: {problem_name}")
    print(f"Specification: {spec_path}")
    print(f"Log path: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-'*80}\n")
    
    # Run the process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream the output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check for errors
    if process.returncode != 0:
        print(f"Error running experiment. Return code: {process.returncode}")
        for line in process.stderr:
            print(line, end='')
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\n{'-'*80}")
    print(f"Experiment completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to: {log_path}")
    print(f"{'-'*80}\n")
    
    return True

def evaluate_experiment(experiment_dir):
    """Evaluate the experiment results using evaluate_results.py."""
    # Construct command
    cmd = ["python", "evaluate_results.py", "--experiment_dir", experiment_dir]
    
    # Execute the command
    print(f"\n{'-'*80}")
    print(f"Evaluating experiment: {experiment_dir}")
    print(f"{'-'*80}\n")
    
    subprocess.run(cmd)
    
    print(f"\n{'-'*80}")
    print(f"Evaluation complete.")
    print(f"{'-'*80}\n")

def run_all_experiments(experiments, results_dir):
    """Run all specified experiments."""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a record of all experiments
    experiment_log = []
    
    # Run each experiment
    for i, experiment in enumerate(experiments):
        experiment_name = experiment.get('name', f'experiment_{i+1}')
        log_path = os.path.join(results_dir, experiment_name)
        
        # Run the experiment
        success = run_experiment(
            problem_name=experiment['problem_name'],
            spec_path=experiment['spec_path'],
            log_path=log_path,
            num_samples=experiment.get('num_samples', 100),
            use_api=experiment.get('use_api', False),
            api_model=experiment.get('api_model', None)
        )
        
        if success:
            # Evaluate the experiment
            evaluate_experiment(log_path)
            
            # Record experiment
            experiment_log.append({
                'experiment_name': experiment_name,
                'status': 'completed',
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': experiment
            })
        else:
            experiment_log.append({
                'experiment_name': experiment_name,
                'status': 'failed',
                'failed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': experiment
            })
        
        # Save experiment log
        with open(os.path.join(results_dir, 'experiment_log.json'), 'w') as f:
            json.dump(experiment_log, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run multiple LLM-SR experiments')
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory to save all experiment results')
    parser.add_argument('--use_api', action='store_true',
                        help='Use OpenAI API for all experiments')
    parser.add_argument('--api_model', type=str, default='gpt-3.5-turbo',
                        help='OpenAI API model to use')
    
    args = parser.parse_args()
    
    # Define experiments
    experiments = [
        {
            'name': 'bactgrow_numpy',
            'problem_name': 'bactgrow',
            'spec_path': './specs/specification_bactgrow_numpy.txt',
            'num_samples': 100,
            'use_api': args.use_api,
            'api_model': args.api_model if args.use_api else None
        },
        # Add more experiments if needed
        # {
        #     'name': 'oscillator1_numpy',
        #     'problem_name': 'oscillator1',
        #     'spec_path': './specs/specification_oscillator1_numpy.txt',
        #     'num_samples': 100,
        #     'use_api': args.use_api,
        #     'api_model': args.api_model if args.use_api else None
        # },
    ]
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.results_dir, f'llmsr_{timestamp}')
    
    print(f"Running {len(experiments)} experiments")
    print(f"Results will be saved to: {results_dir}")
    
    # Run all experiments
    run_all_experiments(experiments, results_dir)
    
    print(f"\n{'-'*80}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_dir}")
    print(f"{'-'*80}\n")


if __name__ == "__main__":
    main()
