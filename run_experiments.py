# run_experiments.py
import subprocess
import os
import random
from typing import List, Dict, Any

# --- Experiment Configuration Space ---
# Define the parameters you want to vary and their possible values.
# This makes it easy to expand the number of experiments.

param_space = {
    'models.seq2seq.generation_params.temperature': [1.0, 1.5, 1.75, 2.0],
    'models.seq2seq.generation_params.min_length': [10, 11, 12, 13, 14, 15],  # 10 [30, 50, 70]
    'models.seq2seq.generation_params.max_length': [16, 17, 18, 19, 20],  # 20 [80, 100, 150]
    'models.seq2seq.generation_params.seed': [1, 10, 42, 100, 1000, 1024, 2048]
    # Add any other parameters you want to explore here
}


def generate_random_configs(num_configs: int) -> List[Dict[str, Any]]:
    """Generates a list of random configurations from the param_space."""
    configs = []
    for i in range(num_configs):
        config_overrides = {}
        # For each parameter, pick a random value from its space
        for param, values in param_space.items():
            config_overrides[param] = random.choice(values)

        # Add a unique run_name and seed for reproducibility
        run_name = f"run_{i + 1:03d}"
        for key, val in config_overrides.items():
            # Create a short name based on key parameters
            if 'sampling' in key:
                run_name += f"_{val.replace(',', '-')}"
            if 'temperature' in key:
                run_name += f"_temp{val}"

        config_overrides['run_name'] = run_name
        config_overrides['active_learning.seed'] = random.randint(1, 10000)  # Unique seed per run

        configs.append(config_overrides)
    return configs


def run_experiment(config_overrides: Dict[str, Any], python_executable: str):
    """Constructs and runs a single experiment as a subprocess."""

    # Start with the base command to run your main script as a module
    cmd = [
        python_executable,
        'main.py',
        '--resume_from_iter',
        '0'
    ]

    # Append the command-line overrides from the config dictionary
    for key, value in config_overrides.items():
        cmd.append(f"{key}={value}")

    run_name = config_overrides.get('run_name', 'unnamed_run')

    print("========================================================")
    print(f"         RUNNING EXPERIMENT: {run_name}")
    print("========================================================")
    print(f"COMMAND: {' '.join(cmd)}\n")

    try:
        # Execute the command. `check=True` will raise an error if the script fails.
        subprocess.run(cmd, check=True)
        print(f"\n--- SUCCESS: Experiment '{run_name}' completed. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- FAILED: Experiment '{run_name}' failed with exit code {e.returncode}. ---\n")
    except Exception as e:
        print(f"\n--- FAILED: An unexpected error occurred while running '{run_name}': {e} ---\n")


def main():
    # Number of experiments you want to run
    num_experiments = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Generate the list of configurations
    experiment_configs = generate_random_configs(num_configs=num_experiments)

    # Or, define specific configurations manually if you prefer
    # manual_configs = [
    #     {'run_name': 'specific_run_1', 'models.seq2seq.sampling_strategies': 'beam', 'models.seq2seq.generation_params.temperature': 0.7, ...},
    #     {'run_name': 'specific_run_2', 'models.seq2seq.sampling_strategies': 'top_p', 'models.seq2seq.generation_params.temperature': 1.2, ...},
    # ]
    # experiment_configs = manual_configs

    # Find the python executable from the current virtual environment
    # This ensures the correct python interpreter is used
    if 'VIRTUAL_ENV' in os.environ:
        python_exe = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python')
    else:
        # Fallback to the python in the current shell's PATH if not in a venv
        python_exe = 'python'
        print("WARNING: Not in a virtual environment. Using system 'python'.")

    # Loop through and run each experiment
    for i, config in enumerate(experiment_configs):
        print(f"Starting experiment {i + 1}/{len(experiment_configs)}...")
        run_experiment(config, python_executable=python_exe)

    print("All experiments have been executed.")


if __name__ == "__main__":
    main()
