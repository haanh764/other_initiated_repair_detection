import sys
from experiment import ExperimentConfig, ExperimentRunner


def run_text_only_experiment():
    """Unimodal: pretrained text embeddings model only."""
    config = ExperimentConfig(
        # Data
        data_file="data/cabb_subset.csv",
        output_dir="./output_text_only",

        # Modalities
        use_text=True,
        use_audio=False,
        use_linguistic=False,
        use_prosodic=False,

        # Model settings
        model_name="pdelobelle/robbert-v2-dutch-base",
        context_mode="none",
        context_window="0",
        use_special_tokens=False,

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=2e-5,

        # Experiment
        experiment_name="pretrained_text_only_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()

def run_audio_only_experiment():
    """Unimodal: pretrained audio embeddings model only."""
    config = ExperimentConfig(
        # Data
        data_file="data/cabb_subset.csv",
        audio_folder="data/audio",
        output_dir="./output_audio_only",

        # Modalities
        use_text=False,
        use_audio=True,
        use_linguistic=False,
        use_prosodic=False,

        # Model settings
        whisper_model_name="openai/whisper-base",

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=0.001,

        # Experiment
        experiment_name="pretrained_audio_only_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_linguistic_only_experiment():
    """Unimodal: handcrafted linguistic features only."""
    config = ExperimentConfig(
        # Data
        data_file="data/combined_handcrafted_features.csv",
        output_dir="./output_linguistic_only",

        # Modalities
        use_text=False,
        use_audio=False,
        use_linguistic=True,
        use_prosodic=False,

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=0.001,

        # Experiment
        experiment_name="linguistic_only_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_prosodic_only_experiment():
    """Unimodal: handcrafted prosodic features only."""
    config = ExperimentConfig(
        # Data
        data_file="data/combined_handcrafted_features.csv",
        output_dir="./output_prosodic_only",

        # Modalities
        use_text=False,
        use_audio=False,
        use_linguistic=False,
        use_prosodic=True,

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=0.001,

        # Experiment
        experiment_name="prosodic_only_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_text_audio_multimodal_experiment():
    """Multimodal: pretrained embeddings text+audio multimodal experiment."""
    config = ExperimentConfig(
        # Data
        data_file="data/cabb_subset.csv",
        audio_folder="data/audio",
        output_dir="./output_pretrained_text_audio",

        # Modalities
        use_text=True,
        use_audio=True,
        use_linguistic=False,
        use_prosodic=False,

        # Model settings
        model_name="pdelobelle/robbert-v2-dutch-base",
        whisper_model_name="openai/whisper-base",
        context_mode="none",
        context_window="0",
        use_special_tokens=False,

        # Fusion settings
        fusion_method="cross_attention",
        fusion_hidden_size=128,
        num_fusion_heads=4,

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=2e-5,

        # Experiment
        experiment_name="text_audio_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_handcrafted_multimodal_experiment():
    """Multimodal: handcrafted linguistic + prosodic features multimodal experiment."""
    config = ExperimentConfig(
        # Data
        data_file="data/combined_handcrafted_features.csv",
        output_dir="./output_handcrafted_multimodal",

        # Modalities
        use_text=False,
        use_audio=False,
        use_linguistic=True,
        use_prosodic=True,

        # Fusion settings
        fusion_method="cross_attention",
        fusion_hidden_size=128,
        num_fusion_heads=4,

        # Training
        batch_size=8,
        max_epochs=20,
        learning_rate=0.001,

        # Experiment
        experiment_name="linguistic_prosodic_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_full_multimodal_experiment():
    """Multimodal (Ours): our proposed full multimodal experiment using all modalities."""
    config = ExperimentConfig(
        # Data
        data_file="data/combined_handcrafted_features.csv",
        audio_folder="data/audio",
        output_dir="./output_full_multimodal",

        # Modalities
        use_text=True,
        use_audio=True,
        use_linguistic=True,
        use_prosodic=True,

        # Model settings
        model_name="pdelobelle/robbert-v2-dutch-base",
        whisper_model_name="openai/whisper-base",
        context_mode="none",
        context_window="0",
        use_special_tokens=False,

        # Fusion settings
        fusion_method="cross_attention",
        fusion_hidden_size=128,
        num_fusion_heads=4,

        # Training
        batch_size=16,
        max_epochs=20,
        learning_rate=2e-5,

        # Experiment
        experiment_name="full_multimodal_no_context_classifier",
        use_kfold=True,
        k_folds=5
    )

    runner = ExperimentRunner(config)
    runner.run()


def run_custom_experiment():
    """Run a custom experiment using command-line arguments"""
    import argparse
    from experiment import create_argument_parser, ExperimentConfig, ExperimentRunner

    parser = create_argument_parser()
    args = parser.parse_args()

    config = ExperimentConfig(**vars(args))
    runner = ExperimentRunner(config)
    runner.run()


# Experiment mapping
EXPERIMENTS = {
    'text': run_text_only_experiment,
    'audio': run_audio_only_experiment,
    'linguistic': run_linguistic_only_experiment,
    'prosodic': run_prosodic_only_experiment,
    'text_audio': run_text_audio_multimodal_experiment,
    'handcrafted': run_handcrafted_multimodal_experiment,
    'full': run_full_multimodal_experiment,
    'custom': run_custom_experiment
}


def main():
    if len(sys.argv) < 2:
        print("Available experiments:")
        for name, func in EXPERIMENTS.items():
            print(f"  {name}: {func.__doc__}")
        print(f"\nUsage: python {sys.argv[0]} <experiment_name>")
        print(f"       python {sys.argv[0]} custom [args...]")
        return

    experiment_name = sys.argv[1]

    if experiment_name not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return

    if experiment_name == 'custom':
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    EXPERIMENTS[experiment_name]()


if __name__ == "__main__":
    main()
