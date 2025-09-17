import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AutoModel
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    F1Score,
)
import torch

class ContextualRepairClassifier(LightningModule):
    """
    Classifier for contextual repair classification using different modalities.
    Supports text embeddings, audio embeddings, handcrafted linguistic, and prosodic features.
    Args:
        base_model_name (str): Name of the base model for text modality.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        max_epochs (int): Maximum number of epochs for training.
        modality (str): Modality to use ('text', 'audio', 'linguistic', 'prosodic').
        audio_dim (int): Dimension of audio features (required for audio modality).
        linguistic_dim (int): Dimension of linguistic features (required for linguistic modality).
        prosodic_dim (int): Dimension of prosodic features (required for prosodic modality).
        freeze_base_model (bool): Whether to freeze the base model parameters.
    """
    def __init__(
        self,
        base_model_name="bert-base-multilingual-cased",
        num_classes=2,
        learning_rate=2e-5,
        warmup_steps=500,
        max_epochs=10,
        modality="text",
        audio_dim=None,
        linguistic_dim=None,
        prosodic_dim=None,
        freeze_base_model=False,
        fusion_method="cross_attention",
        fusion_hidden_size=128,
        num_fusion_heads=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modality = modality
        self.fusion_method = fusion_method

        self.use_text = "text" in modality
        self.use_audio = "audio" in modality
        self.use_linguistic = "linguistic" in modality
        self.use_prosodic = "prosodic" in modality
        self.is_multimodal = "+" in modality

        # Text modality components
        if self.use_text:
            self.base_model = AutoModel.from_pretrained(base_model_name)
            self.text_dim = self.base_model.config.hidden_size

            if freeze_base_model:
                for param in self.base_model.parameters():
                    param.requires_grad = False

        # Audio modality components
        if self.use_audio:
            if audio_dim is None:
                raise ValueError("audio_dim must be specified for audio modality")
            self.audio_dim = audio_dim

        # Linguistic modality components
        if self.use_linguistic:
            if linguistic_dim is None:
                raise ValueError("linguistic_dim must be specified for linguistic modality")
            self.linguistic_dim = linguistic_dim

        # Prosodic modality components
        if self.use_prosodic:
            if prosodic_dim is None:
                raise ValueError("prosodic_dim must be specified for prosodic modality")
            self.prosodic_dim = prosodic_dim

        if not self.is_multimodal:
            # Define a simple classifier for unimodal cases
            if self.use_text:
                self.hidden_size = self.text_dim
            elif self.use_audio:
                self.hidden_size = self.audio_dim
            elif self.use_linguistic:
                self.hidden_size = self.linguistic_dim
            elif self.use_prosodic:
                self.hidden_size = self.prosodic_dim

            if self.modality == "text":
                self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size, num_classes)
                )
            elif self.modality == "audio":
                self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, num_classes)
                )
            elif self.modality in ["linguistic", "prosodic"]:
                self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes)
                )
        else:
            self.fusion_hidden_size = fusion_hidden_size
            if self.use_text:
                self.text_projection = nn.Linear(self.text_dim, self.fusion_hidden_size)

            if self.use_audio:
                self.audio_projection = nn.Linear(self.audio_dim, self.fusion_hidden_size)

            if self.use_linguistic:
                self.linguistic_projection = nn.Linear(self.linguistic_dim, self.fusion_hidden_size)

            if self.use_prosodic:
                self.prosodic_projection = nn.Linear(self.prosodic_dim, self.fusion_hidden_size)

            if self.fusion_method == "cross_attention":
                self.cross_attention_layers = nn.ModuleList()
                modality_count = sum([self.use_text, self.use_audio, self.use_linguistic, self.use_prosodic])
                for _ in range(modality_count - 1):
                    self.cross_attention_layers.append(
                        nn.MultiheadAttention(
                            embed_dim=fusion_hidden_size,
                            num_heads=num_fusion_heads,
                            batch_first=True
                        )
                    )
                self.classifier = nn.Sequential(
                    nn.Linear(fusion_hidden_size, fusion_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(fusion_hidden_size, num_classes)
                )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self._setup_metrics()

    def _setup_metrics(self):
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_f1_macro = F1Score(task="binary", average="macro")
        self.val_f1_macro = F1Score(task="binary", average="macro")
        self.test_f1_macro = F1Score(task="binary", average="macro")

        self.train_f1_micro = F1Score(task="binary", average="micro")
        self.val_f1_micro = F1Score(task="binary", average="micro")
        self.test_f1_micro = F1Score(task="binary", average="micro")

        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

    def extract_features(self, batch):
        """
        Extract features from the batch based on the given modality.
        """
        features = {}
        if self.use_text:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            features['text'] = outputs.last_hidden_state[:, 0, :]
        if self.use_audio:
            features['audio'] = batch['audio_features']
        if self.use_linguistic:
            features['linguistic'] = batch['linguistic_features']
        if self.use_prosodic:
            features['prosodic'] = batch['prosodic_features']
        return features

    def fuse_features(self, features):
        """
        Fuse features from different modalities using cross-attention.
        """
        projected_features = []
        if self.use_text:
            projected_features.append(self.text_projection(features['text']))
        if self.use_audio:
            projected_features.append(self.audio_projection(features['audio']))
        if self.use_linguistic:
            projected_features.append(self.linguistic_projection(features['linguistic']))
        if self.use_prosodic:
            projected_features.append(self.prosodic_projection(features['prosodic']))

        fused_feature = projected_features[0]
        for i, attn_layer in enumerate(self.cross_attention_layers):
            next_feature = projected_features[i + 1]

            if len(fused_feature.shape) == 2:
                fused_feature = fused_feature.unsqueeze(1)  # Add seq_len dimension
            if len(next_feature.shape) == 2:
                next_feature = next_feature.unsqueeze(1)

            attn_output, _ = attn_layer(
                query=fused_feature,
                key=next_feature,
                value=next_feature
            )
            fused_feature = fused_feature + attn_output

        if len(fused_feature.shape) > 2:
            fused_feature = fused_feature.squeeze(1)
        return fused_feature

    def forward(self, batch):
        if not self.is_multimodal:
            # Unimodal case
            if self.modality == "text":
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                features = outputs.last_hidden_state[:, 0, :]
            elif self.modality == "audio":
                features = batch['audio_features']
            elif self.modality == "linguistic":
                features = batch['linguistic_features']
            elif self.modality == "prosodic":
                features = batch['prosodic_features']
        else:
            # Multimodal case
            features_dict = self.extract_features(batch)
            features = self.fuse_features(features_dict)

        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels']

        loss = self.loss_fn(logits[:, 1], labels.float())

        preds_prob = torch.sigmoid(logits[:, 1])
        preds = (preds_prob > 0.5).long()

        self.train_acc(preds, labels)
        self.train_f1_macro(preds, labels)
        self.train_f1_micro(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1_macro", self.train_f1_macro, on_step=False, on_epoch=True)
        self.log("train/f1_micro", self.train_f1_micro, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels']

        loss = self.loss_fn(logits[:, 1], labels.float())

        preds_prob = torch.sigmoid(logits[:, 1])
        preds = (preds_prob > 0.5).long()

        self.val_acc(preds, labels)
        self.val_f1_macro(preds, labels)
        self.val_f1_micro(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True)
        self.log("val/f1_micro", self.val_f1_micro, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels']

        loss = self.loss_fn(logits[:, 1], labels.float())

        preds_prob = torch.sigmoid(logits[:, 1])
        preds = (preds_prob > 0.5).long()

        self.test_acc(preds, labels)
        self.test_f1_macro(preds, labels)
        self.test_f1_micro(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)
        self.log("test/f1_micro", self.test_f1_micro, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits[:, 1])
        preds = (probs > 0.5).long()

        return {
            'probabilities': probs,
            'predictions': preds,
            'conv_id': batch['conv_id'],
            'onset_msec': batch['onset_msec'],
            'offset_msec': batch['offset_msec']
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                               verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            }
        }
