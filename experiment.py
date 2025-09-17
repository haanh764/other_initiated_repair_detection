import os
import argparse
import torch
import pandas as pd
import numpy as np
import wandb
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import train_test_split, KFold

from transformers import AutoTokenizer, AutoModel, WhisperProcessor, WhisperModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from dataset import ContextualConversationDataset
from model import ContextualRepairClassifier
from utils import int_or_max


class ExperimentConfig:
    """Configuration class for experiments"""

    def __init__(self, **kwargs):
        # Data arguments
        self.data_file: str = kwargs.get('data_file')
        self.audio_folder: Optional[str] = kwargs.get('audio_folder')
        self.label_column: str = kwargs.get('label_column', 'label')
        self.output_dir: str = kwargs.get('output_dir', './output')
        self.num_classes: int = kwargs.get('num_classes', 2)

        # Modality settings
        self.use_text: bool = kwargs.get('use_text', False)
        self.use_audio: bool = kwargs.get('use_audio', False)
        self.use_linguistic: bool = kwargs.get('use_linguistic', False)
        self.use_prosodic: bool = kwargs.get('use_prosodic', False)

        # Model arguments
        self.model_name: str = kwargs.get('model_name', 'pdelobelle/robbert-v2-dutch-base')
        self.whisper_model_name: str = kwargs.get('whisper_model_name', 'openai/whisper-base')
        self.max_length: int = kwargs.get('max_length', 512)
        self.context_window = kwargs.get('context_window', 3)
        self.context_mode: str = kwargs.get('context_mode', 'both')
        self.use_special_tokens: bool = kwargs.get('use_special_tokens', False)
        self.freeze_base_model: bool = kwargs.get('freeze_base_model', False)

        # Fusion arguments
        self.fusion_method: str = kwargs.get('fusion_method', 'cross_attention')
        self.fusion_hidden_size: int = kwargs.get('fusion_hidden_size', 128)
        self.num_fusion_heads: int = kwargs.get('num_fusion_heads', 4)

        # Training arguments
        self.batch_size: int = kwargs.get('batch_size', 16)
        self.learning_rate: float = kwargs.get('learning_rate', 2e-5)
        self.max_epochs: int = kwargs.get('max_epochs', 20)
        self.warmup_steps: int = kwargs.get('warmup_steps', 500)
        self.patience: int = kwargs.get('patience', 3)
        self.num_workers: int = kwargs.get('num_workers', 4)

        # Experiment settings
        self.experiment_name: str = kwargs.get('experiment_name', 'experiment')
        self.use_kfold: bool = kwargs.get('use_kfold', False)
        self.k_folds: int = kwargs.get('k_folds', 5)
        self.project_name: str = kwargs.get('project_name', 'other-repair-initiation-classifier')

    @property
    def modality_string(self) -> str:
        """Create string containing active modalities"""
        modalities = []
        if self.use_text:
            modalities.append("text")
        if self.use_audio:
            modalities.append("audio")
        if self.use_linguistic:
            modalities.append("linguistic")
        if self.use_prosodic:
            modalities.append("prosodic")
        return "+".join(modalities) if modalities else ""

    @property
    def is_multimodal(self) -> bool:
        """Check if this is a multimodal experiment"""
        return "+" in self.modality_string

    def validate(self):
        """Validate configuration"""
        if not any([self.use_text, self.use_audio, self.use_linguistic, self.use_prosodic]):
            raise ValueError("At least one modality must be enabled")

        if self.use_audio and not self.audio_folder:
            raise ValueError("Audio folder must be specified when using audio modality")


class ModelComponents:
    """Initialized model components"""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.base_model: Optional[AutoModel] = None
        self.audio_processor: Optional[WhisperProcessor] = None
        self.audio_model: Optional[WhisperModel] = None
        self.max_length: Optional[int] = None


class ExperimentRunner:
    """Main experiment runner"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.validate()
        self.components = ModelComponents()

    def _initialize_components(self):
        """Initialize model components"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.use_text:
            print("Initializing text model components...")
            self.components.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.components.base_model = AutoModel.from_pretrained(self.config.model_name)
            self.components.max_length = self.components.base_model.config.max_position_embeddings

        if self.config.use_audio:
            print("Initializing audio model components...")
            self.components.audio_processor = WhisperProcessor.from_pretrained(self.config.whisper_model_name)
            self.components.audio_model = WhisperModel.from_pretrained(self.config.whisper_model_name)
            self.components.audio_model = self.components.audio_model.to(device)

    def _create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[
        ContextualConversationDataset, ContextualConversationDataset, ContextualConversationDataset]:
        """Create datasets for train, validation, and test"""

        dataset_kwargs = {
            'tokenizer': self.components.tokenizer,
            'max_length': self.components.max_length,
            'context_window': self.config.context_window,
            'context_mode': self.config.context_mode,
            'use_special_tokens': self.config.use_special_tokens,
            'use_text': self.config.use_text,
            'use_audio': self.config.use_audio,
            'use_linguistic': self.config.use_linguistic,
            'use_prosodic': self.config.use_prosodic,
            'audio_folder': self.config.audio_folder,
            'whisper_model': self.components.audio_model,
            'whisper_pprocessor': self.components.audio_processor
        }

        train_dataset = ContextualConversationDataset(train_df, **dataset_kwargs)
        val_dataset = ContextualConversationDataset(val_df, **dataset_kwargs)
        test_dataset = ContextualConversationDataset(test_df, **dataset_kwargs)
        return train_dataset, val_dataset, test_dataset

    def _create_data_loaders(self, train_dataset, val_dataset, test_dataset) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers
        )
        return train_loader, val_loader, test_loader

    def _get_feature_dimensions(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get feature dimensions for each modality"""
        dimensions = {}

        if self.config.use_audio:
            dimensions['audio'] = self.components.audio_model.config.d_model

        if self.config.use_linguistic:
            linguistic_cols = [col for col in df.columns if col.startswith('ling_')]
            dimensions['linguistic'] = len(linguistic_cols)
            print(f"Found {dimensions['linguistic']} linguistic features")

        if self.config.use_prosodic:
            prosodic_cols = [col for col in df.columns if col.startswith('pros_')]
            dimensions['prosodic'] = len(prosodic_cols)
            print(f"Found {dimensions['prosodic']} prosodic features")
        return dimensions

    def _create_model(self, feature_dims: Dict[str, int]) -> ContextualRepairClassifier:
        """Create model for repair initiation detection"""

        model_kwargs = {
            'num_classes': self.config.num_classes,
            'learning_rate': self.config.learning_rate,
            'warmup_steps': self.config.warmup_steps,
            'max_epochs': self.config.max_epochs,
            'modality': self.config.modality_string,
            'freeze_base_model': self.config.freeze_base_model
        }

        if self.config.use_text:
            model_kwargs['base_model_name'] = self.config.model_name

        if self.config.use_audio:
            model_kwargs['audio_dim'] = feature_dims['audio']

        if self.config.use_linguistic:
            model_kwargs['linguistic_dim'] = feature_dims['linguistic']

        if self.config.use_prosodic:
            model_kwargs['prosodic_dim'] = feature_dims['prosodic']

        if self.config.is_multimodal:
            model_kwargs.update({
                'fusion_method': self.config.fusion_method,
                'fusion_hidden_size': self.config.fusion_hidden_size,
                'num_fusion_heads': self.config.num_fusion_heads
            })
        return ContextualRepairClassifier(**model_kwargs)

    def _create_callbacks(self, fold: Optional[int] = None):
        """Create training callbacks"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"fold_{fold}_checkpoints" if fold is not None else "checkpoints"
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=1 if fold is not None else 3
        )

        early_stop_callback = EarlyStopping(
            monitor="val/acc",
            patience=self.config.patience,
            mode="max"
        )
        return [checkpoint_callback, early_stop_callback]

    def _save_model_and_components(self, trainer: Trainer, fold: Optional[int] = None):
        """Save model checkpoint and tokenizer"""
        save_dir = os.path.join(
            self.config.output_dir,
            f"fold_{fold}_model" if fold is not None else "final_model"
        )
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_checkpoint(os.path.join(save_dir, "model.ckpt"))

        if self.components.tokenizer is not None:
            self.components.tokenizer.save_pretrained(save_dir)

    def _train_single_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame, fold: Optional[int] = None,
                           wandb_logger: Optional[WandbLogger] = None) -> Dict[str, float]:

        # Create dataset loaders
        train_dataset, val_dataset, test_dataset = self._create_datasets(train_df, val_df, test_df)
        train_loader, val_loader, test_loader = self._create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
        feature_dims = self._get_feature_dimensions(train_df)
        model = self._create_model(feature_dims)
        callbacks = self._create_callbacks(fold)

        trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=self.config.max_epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            num_sanity_val_steps=0
        )

        trainer.fit(model, train_loader, val_loader)
        test_results = trainer.test(model, test_loader)[0]

        self._save_model_and_components(trainer, fold)

        del model, trainer, train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache()
        return test_results

    def run_kfold_experiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Run k-fold cross-validation experiment"""
        kfold = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        indices = np.arange(len(df))

        fold_results = {}
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
            print(f"\n{'=' * 50}\nFold {fold + 1}/{self.config.k_folds}\n{'=' * 50}")

            if wandb.run:
                wandb.finish()

            wandb.init(
                project=self.config.project_name,
                name=f"{self.config.experiment_name}_fold_{fold + 1}",
                config={**vars(self.config), 'fold': fold + 1}
            )
            wandb_logger = WandbLogger(experiment=wandb.run)

            train_val_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.2, random_state=42,
                stratify=train_val_df[self.config.label_column]
            )
            print(f"Fold {fold + 1} - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            test_results = self._train_single_fold(train_df, val_df, test_df, fold + 1, wandb_logger)

            for metric, value in test_results.items():
                if metric not in fold_results:
                    fold_results[metric] = []
                fold_results[metric].append(value)

        results_summary = {}
        for metric, values in fold_results.items():
            if values:
                results_summary[f"{metric}_mean"] = np.mean(values)
                results_summary[f"{metric}_std"] = np.std(values)

        if wandb.run:
            wandb.finish()

        wandb.init(
            project=self.config.project_name,
            name=f"{self.config.experiment_name}_summary",
            config=vars(self.config)
        )
        wandb.log(results_summary)
        wandb.finish()

        self._save_kfold_results(results_summary)
        return results_summary

    def run_single_experiment(self, df: pd.DataFrame) -> Dict[str, float]:
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42,
            stratify=df[self.config.label_column]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42,
            stratify=temp_df[self.config.label_column]
        )
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        wandb_logger = WandbLogger(
            project=self.config.project_name,
            name=self.config.experiment_name,
            log_model=True
        )
        wandb_logger.log_hyperparams(vars(self.config))
        test_results = self._train_single_fold(train_df, val_df, test_df, None, wandb_logger)
        results_df = pd.DataFrame([test_results])
        results_df.to_csv(os.path.join(self.config.output_dir, "test_results.csv"), index=False)
        wandb.finish()
        return test_results

    def _save_kfold_results(self, results_summary: Dict[str, float]):
        """Save k-fold results to CSV"""
        results_dict = {'metric': [], 'mean': [], 'std': []}

        for key, value in results_summary.items():
            if key.endswith('_mean'):
                metric_name = key[:-5]
                mean_value = value
                std_value = results_summary.get(f"{metric_name}_std", float('nan'))

                results_dict['metric'].append(metric_name)
                results_dict['mean'].append(mean_value)
                results_dict['std'].append(std_value)

        results_df = pd.DataFrame(results_dict)
        results_path = os.path.join(self.config.output_dir, "kfold_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"K-fold results saved to {results_path}")

    def run(self):
        """Running experiments"""
        print(f"Starting experiment: {self.config.experiment_name}")
        print(f"Modalities used: {self.config.modality_string}")

        # Load data
        df = pd.read_csv(self.config.data_file)
        print(f"Loaded dataset with {len(df)} samples")

        # Initialize components
        self._initialize_components()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Run experiment
        if self.config.use_kfold:
            results = self.run_kfold_experiment(df)
            print("\n" + "=" * 50)
            print(f"K-Fold Cross-Validation Results (k={self.config.k_folds}):")
            print("=" * 50)
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
        else:
            results = self.run_single_experiment(df)
            print("\n" + "=" * 50)
            print("Test Results:")
            print("=" * 50)
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for experiments"""
    parser = argparse.ArgumentParser(description="Experiments for multimodal classifier for OIR detection")

    # K-Fold configuration
    parser.add_argument("--use_kfold", action="store_true", help="Use k-fold cross-validation")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")

    # Data configuration
    parser.add_argument("--data_file", type=str, required=True, help="Path to data CSV file")
    parser.add_argument("--audio_folder", type=str, help="Path to audio files directory")
    parser.add_argument("--label_column", type=str, default="label", help="Name of label column")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")

    # Modality configuration
    parser.add_argument("--use_text", action="store_true", help="Use text pretrained embeddings (transcription)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio pretrained embeddings")
    parser.add_argument("--use_linguistic", action="store_true", help="Use handcrafted linguistic features")
    parser.add_argument("--use_prosodic", action="store_true", help="Use handcrafted prosodic features")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="pdelobelle/robbert-v2-dutch-base",
                        help="Pretrained text model name")
    parser.add_argument("--whisper_model_name", type=str, default="openai/whisper-base",
                        help="Whisper model for audio processing")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--context_window", type=int_or_max, default=3,
                        help="Micro context window size or 'max'")
    parser.add_argument("--context_mode", type=str, default="both",
                        choices=["none", "past", "future", "both"], help="Context mode")
    parser.add_argument("--use_special_tokens", action="store_true",
                        help="Use special tokens to concatenate context segments")
    parser.add_argument("--freeze_base_model", action="store_true",
                        help="Freeze base model parameters")

    # Fusion configuration
    parser.add_argument("--fusion_method", type=str, default="cross_attention",
                        choices=["cross_attention", "concat"], help="Method for modality fusion")
    parser.add_argument("--fusion_hidden_size", type=int, default=128,
                        help="Hidden size for fusion layer")
    parser.add_argument("--num_fusion_heads", type=int, default=4,
                        help="Number of attention heads for fusion")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--project_name", type=str, default="other-repair-initiation-classifier",
                        help="WandB project name")
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    config = ExperimentConfig(**vars(args))
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
