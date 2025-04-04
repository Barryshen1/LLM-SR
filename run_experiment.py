import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator
from llmsr import code_manipulation

class ExperimentProfiler:
    """Custom profiler to track equation generation and evaluation during experiments."""
    
    def __init__(self, log_dir, max_samples=100):
        """
        Initialize experiment profiler
        
        Args:
            log_dir: Directory to save experiment logs
            max_samples: Maximum number of samples to collect
        """
        self.log_dir = log_dir
        self.max_samples = max_samples
        self.results_dir = os.path.join(log_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.samples_collected = 0
        self.best_score = float('-inf')
        self.best_function = None
        self.all_functions = []
        self.start_time = time.time()
        
        # Save experiment metadata
        self.metadata = {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'max_samples': max_samples
        }
        with open(os.path.join(self.log_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_function(self, function):
        """Register a function and its evaluation results."""
        if self.samples_collected >= self.max_samples:
            return False
            
        self.samples_collected += 1
        
        # Extract information
        score = function.score
        function_str = str(function).strip()
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        
        # Track best function
        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_function = function_str
        
        # Store function data
        function_data = {
            'sample_num': self.samples_collected,
            'function': function_str,
            'score': score,
            'sample_time': sample_time,
            'evaluate_time': evaluate_time,
            'elapsed_time': time.time() - self.start_time
        }
        self.all_functions.append(function_data)
        
        # Save individual sample
        with open(os.path.join(self.results_dir, f'sample_{self.samples_collected:03d}.json'), 'w') as f:
            json.dump(function_data, f, indent=2)
            
        # Print progress
        print(f"\n=== Sample {self.samples_collected}/{self.max_samples} ===")
        print(f"Score: {score}")
        print(f"Elapsed time: {time.time() - self.start_time:.2f}s")
        
        # Save summary after each update
        self._save_summary()
        
        return self.samples_collected < self.max_samples
    
    def _save_summary(self):
        """Save experiment summary to file."""
        summary = {
            'total_samples': self.samples_collected,
            'elapsed_time': time.time() - self.start_time,
            'best_score': self.best_score,
            'best_function': self.best_function,
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save all samples data
        with open(os.path.join(self.log_dir, 'all_samples.json'), 'w') as f:
            json.dump(self.all_functions, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run LLM-SR experiment with custom sample limit')
    parser.add_argument('--problem_name', type=str, default="bactgrow", 
                        help='Name of the benchmark problem')
    parser.add_argument('--spec_path', type=str, default="./specs/specification_bactgrow_numpy.txt", 
                        help='Path to the specification file')
    parser.add_argument('--log_path', type=str, default="./experiments/bactgrow_experiment", 
                        help='Path to save experiment logs')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to generate')
    parser.add_argument('--use_api', action='store_true', 
                        help='Use OpenAI API instead of local LLM')
    parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo", 
                        help='Model to use with OpenAI API')
    
    args = parser.parse_args()
    
    # Create experiment directory if it doesn't exist
    os.makedirs(args.log_path, exist_ok=True)
    
    # Load prompt specification
    with open(args.spec_path, encoding="utf-8") as f:
        specification = f.read()
    
    # Load dataset
    df = pd.read_csv(f'./data/{args.problem_name}/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    
    # Convert to torch tensor if using torch specification
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}
    
    # Initialize config
    llm_config = config.Config(
        use_api=args.use_api,
        api_model=args.api_model,
    )
    
    # Set up the class config
    class_config = config.ClassConfig(
        llm_class=sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox
    )
    
    # Initialize custom profiler
    profiler = ExperimentProfiler(args.log_path, max_samples=args.num_samples)
    
    # Extract function names from specification
    function_to_evolve, function_to_run = pipeline._extract_function_names(specification)
    
    # Create the template program
    template = code_manipulation.text_to_program(specification)
    
    # Initialize experience buffer
    database = pipeline.buffer.ExperienceBuffer(
        llm_config.experience_buffer,
        template,
        function_to_evolve
    )
    
    # Create evaluators
    evaluators = []
    for _ in range(llm_config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            dataset,
            timeout_seconds=llm_config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))
    
    # Evaluate initial function
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)
    
    # Create samplers
    samplers = [
        sampler.Sampler(
            database,
            evaluators,
            llm_config.samples_per_prompt,
            max_sample_nums=args.num_samples,
            llm_class=class_config.llm_class,
            config=llm_config
        )
    ]
    
    # Define a custom sample function that stops after reaching the sample limit
    def custom_sample(sampler_instance, profiler):
        while True:
            prompt = database.get_prompt()
            
            reset_time = time.time()
            samples = sampler_instance._llm.draw_samples(prompt.code, llm_config)
            sample_time = (time.time() - reset_time) / sampler_instance._samples_per_prompt
            
            for sample in samples:
                sampler_instance._global_sample_nums_plus_one()
                cur_global_sample_nums = sampler_instance._get_global_sample_nums()
                
                chosen_evaluator = np.random.choice(evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    profiler=profiler,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                
                # Check if we've reached the sample limit
                if cur_global_sample_nums >= args.num_samples:
                    return
    
    # Run the sampler
    custom_sample(samplers[0], profiler)
    
    print(f"\n=== Experiment complete ===")
    print(f"Generated {profiler.samples_collected} samples")
    print(f"Best score: {profiler.best_score}")
    print(f"Total time: {time.time() - profiler.start_time:.2f} seconds")
    print(f"Results saved to: {args.log_path}")


if __name__ == "__main__":
    main()
