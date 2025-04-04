import os
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator

def run_experiment(problem_name, spec_path, log_path, use_api=False, api_model="gpt-3.5-turbo", max_sample_nums=10000):
    """Run a single LLM-SR experiment."""
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config_obj = config.Config(use_api=use_api, api_model=api_model)
    
    # Load prompt specification
    with open(os.path.join(spec_path), encoding="utf-8") as f:
        specification = f.read()
    
    # Load dataset
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}
    
    # Run the pipeline
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config_obj,
        max_sample_nums=max_sample_nums,
        class_config=class_config,
        log_dir=log_path,
    )
    
    return log_path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--spec_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default="./logs/experiment")
    parser.add_argument('--problem_name', type=str, required=True)
    parser.add_argument('--max_sample_nums', type=int, default=10000)
    
    args = parser.parse_args()
    
    run_experiment(
        problem_name=args.problem_name,
        spec_path=args.spec_path,
        log_path=args.log_path,
        use_api=args.use_api,
        api_model=args.api_model,
        max_sample_nums=args.max_sample_nums
    )
