import os
import subprocess
import sys
import argparse

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'torch': 'torch',
        'scipy': 'scipy',
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package} is NOT installed")
    
    return missing_packages

def install_packages(packages):
    """Install missing packages using pip."""
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies installed successfully!")

def check_project_structure():
    """Check if the project structure is correct."""
    required_dirs = ['specs', 'data', 'llmsr']
    required_files = ['main.py']
    
    missing_items = []
    
    # Check directories
    for directory in required_dirs:
        if not os.path.isdir(directory):
            missing_items.append(f"Directory '{directory}' is missing")
    
    # Check files
    for file in required_files:
        if not os.path.isfile(file):
            missing_items.append(f"File '{file}' is missing")
    
    # Check specific benchmark data
    if not os.path.isdir('data/bactgrow'):
        missing_items.append("Benchmark 'bactgrow' is missing from data directory")
    elif not os.path.isfile('data/bactgrow/train.csv'):
        missing_items.append("File 'train.csv' is missing for 'bactgrow' benchmark")
    
    return missing_items

def setup_experiment_dirs():
    """Create necessary directories for experiments."""
    dirs_to_create = [
        'experiments',
        'experiment_results',
    ]
    
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def copy_experiment_scripts():
    """Copy experiment scripts to the correct location."""
    # Ensure the scripts are in the current directory
    script_files = ['run_experiment.py', 'evaluate_results.py', 'run_experiments.py']
    
    for script in script_files:
        if not os.path.isfile(script):
            print(f"Warning: Could not find {script} in the current directory.")
        else:
            print(f"Found script: {script}")
    
    print("All experiment scripts are ready.")

def main():
    parser = argparse.ArgumentParser(description='Setup LLM-SR experiment environment')
    parser.add_argument('--install', action='store_true', 
                        help='Install missing dependencies')
    args = parser.parse_args()
    
    print("=== Checking dependencies ===")
    missing_packages = check_dependencies()
    
    if missing_packages:
        if args.install:
            print("\n=== Installing missing packages ===")
            install_packages(missing_packages)
        else:
            print("\nMissing packages found. Run with --install to install them.")
            print("Command: python setup_experiment.py --install")
    else:
        print("All required packages are installed.")
    
    print("\n=== Checking project structure ===")
    missing_items = check_project_structure()
    
    if missing_items:
        print("\nWarning: The following items are missing:")
        for item in missing_items:
            print(f"  - {item}")
        print("\nPlease make sure you're in the correct directory or clone the repository again.")
    else:
        print("Project structure looks good!")
    
    print("\n=== Setting up experiment directories ===")
    setup_experiment_dirs()
    
    print("\n=== Checking experiment scripts ===")
    copy_experiment_scripts()
    
    print("\n=== Setup complete ===")
    print("You can now run experiments using:")
    print("  python run_experiments.py")
    print("Or run a single experiment using:")
    print("  python run_experiment.py --problem_name bactgrow --spec_path ./specs/specification_bactgrow_numpy.txt --log_path ./experiments/bactgrow_test --num_samples 100")


if __name__ == "__main__":
    main()
