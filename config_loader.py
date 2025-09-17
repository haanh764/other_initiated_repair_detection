"""
Configuration loader for running experiments using config from YAML files.
"""

import yaml
import argparse
from typing import Dict, Any
from experiment import ExperimentConfig, ExperimentRunner


def load_config(config_file: str, experiment_name: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)

    if experiment_name:
        if experiment_name not in configs:
            available = list(configs.keys())
            raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")
        return configs[experiment_name]
    return configs


def run_from_config(config_file: str, experiment_name: str, overrides: Dict[str, Any] = None):
    """Run experiment from YAML config files, with optional overrides"""
    config_dict = load_config(config_file, experiment_name)

    if overrides:
        config_dict.update(overrides)

    config = ExperimentConfig(**config_dict)
    runner = ExperimentRunner(config)
    runner.run()


def parse_overrides(override_args: list) -> Dict[str, Any]:
    """Parse command-line override arguments in key=value format"""
    overrides = {}

    for arg in override_args:
        if '=' not in arg:
            continue

        key, value = arg.split('=', 1)

        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)

        overrides[key] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments from YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Run text_only experiment from config file
              python config_loader.py experiment_configs.yaml text_only
            
              # Run with overrides
              python config_loader.py experiment_configs.yaml text_only batch_size=32 learning_rate=1e-4
            
              # List available experiments
              python config_loader.py experiment_configs.yaml --list
        """
    )

    parser.add_argument("config_file", help="Path to YAML config file")
    parser.add_argument("experiment_name", nargs='?', help="Experiment name to run")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("overrides", nargs='*', help="(Optional) Override config params in key=value format")

    args = parser.parse_args()

    if args.list:
        configs = load_config(args.config_file)
        print("Available experiments:")
        for name, config in configs.items():
            if isinstance(config, dict) and 'experiment_name' in config:
                desc = config.get('experiment_name', name)
                modalities = []
                if config.get('use_text'): modalities.append('text')
                if config.get('use_audio'): modalities.append('audio')
                if config.get('use_linguistic'): modalities.append('linguistic')
                if config.get('use_prosodic'): modalities.append('prosodic')
                modality_str = '+'.join(modalities) if modalities else 'none'
                print(f"  {name}: {desc} ({modality_str})")
        return

    if not args.experiment_name:
        parser.print_help()
        return

    overrides = parse_overrides(args.overrides)
    run_from_config(args.config_file, args.experiment_name, overrides)


if __name__ == "__main__":
    main()
