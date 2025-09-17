import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os
from pydub import AudioSegment
import librosa
import numpy as np


class ContextualConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, context_window=3, context_mode="both", use_special_tokens=True,
                 use_text=True, use_audio=False, use_linguistic=False, use_prosodic=False, audio_folder=None, whisper_model=None,
                 whisper_pprocessor=None):
        """
        Dataset for contextual conversation repair analysis with flexible context handling

        Args:
            dataframe: Input dataframe with conversation data
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
            context_window: Number of segments to include from past/future (if available), or "max" for as many as possible
            context_mode: How to handle context - "none", "past", "future", "both"
            use_special_tokens: Whether to use [SEP] tokens between segments
            use_text: Whether to use text modality
            use_audio: Whether to use audio modality
            use_linguistic: Whether to use linguistic features
            use_prosodic: Whether to use prosodic features
            audio_folder: Path to the folder containing audio files (if modality is "audio")
            whisper_model: Whisper model for audio processing (if modality is "audio")

        Example usage:
            dataset = ContextualConversationDataset(
                dataframe=df,
                tokenizer=tokenizer,
                max_length=512,
                context_window=3,
                context_mode="both",
                use_special_tokens=True,
                use_text=True,
                use_audio=True,
                audio_folder="/path/to/audio/folder",
                whisper_model=whisper_model
            )
        """
        self.df = dataframe
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_linguistic = use_linguistic
        self.use_prosodic = use_prosodic

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not any([use_text, use_audio, use_linguistic, use_prosodic]):
            raise ValueError("At least one modality must be enabled")

        # Text-specific attributes
        # self.use_text = self.modality in ["text", "multimodal"]
        if self.use_text:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for text and multimodal modality")
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.context_window = context_window
            self.context_mode = context_mode
            self.use_special_tokens = use_special_tokens
            self.sep_token = "[SEP]" if use_special_tokens else " "

            valid_modes = ["none", "past", "future", "both"]
            if self.context_mode not in valid_modes:
                raise ValueError(f"Context mode must be one of {valid_modes}, got {self.context_mode}")

            print(f"Context mode: {context_mode}, window: {context_window}")
            self.prepare_text_inputs()

        # Audio-specific attributes
        # self.use_audio = self.modality in ["audio", "multimodal"]
        if self.use_audio:
            if audio_folder is None or whisper_model is None:
                raise ValueError("Audio folder and Whisper model must be provided for audio and multimodal modality")
            self.audio_folder = audio_folder
            self.whisper_model = whisper_model
            self.whisper_processor = whisper_pprocessor
            self.audio_features = []
            print("Loading audio features from folder: {audio_folder}")
            self.prepare_audio_embedding_input()

        # Linguistic features
        self.use_linguistic = use_linguistic
        if self.use_linguistic:
            print("Loading linguistic features from dataframe")
            self.prepare_linguistic_features()

        # Prosodic features
        self.use_prosodic = use_prosodic
        if self.use_prosodic:
            print("Loading prosodic features from dataframe")
            self.prepare_prosodic_features()

        # Prepare labels
        self.labels = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preparing labels"):
            if 'label' in row:
                self.labels.append(row['label'])
            else:
                self.labels.append(-1)


    def prepare_text_inputs(self):
        """Prepare inputs for each segment with incorporated context"""
        self.text_inputs = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preparing text inputs"):
            context = self.build_context(row)

            encoded = self.tokenizer(
                context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            self.text_inputs.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })

    def prepare_audio_embedding_input(self):
        """Extract audio features for all instances"""
        self.audio_features = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting audio features"):
            features = self.extract_audio_features(row)
            if features is None:
                features = torch.zeros(self.whisper_model.encoder.config.d_model, dtype=torch.float)
            self.audio_features.append(features)

    def extract_audio_features(self, row):
        """Extract audio features for a single instance. Use audio file corresponding for current speaker (if available),
        otherwise use combined audio file."""
        try:
            speaker = row['speaker']
            pairnr = row['pairnr']

            speaker_audio_path = os.path.join(self.audio_folder, f"pair{pairnr}_synced_pp{speaker}.wav")
            combined_audio_path = os.path.join(self.audio_folder, f"pair{pairnr}_synced.wav")

            if os.path.exists(speaker_audio_path):
                audio_path = speaker_audio_path
            elif os.path.exists(combined_audio_path):
                audio_path = combined_audio_path
            else:
                print(f"No audio file found for {pairnr}")
                return None

            # Load audio file
            audio = AudioSegment.from_file(audio_path)

            # Extract segment based on timestamps
            start_ms = int(row['onset_msec'])
            end_ms = int(row['offset_msec'])
            segment = audio[start_ms:end_ms]

            segment_sampled = np.array(segment.get_array_of_samples())

            # Convert to float and normalize as required by WhisperProcessor
            if segment.sample_width == 2:  # 16-bit audio
                segment_sampled = segment_sampled.astype(np.float32) / 32768.0
            elif segment.sample_width == 4:  # 32-bit audio
                segment_sampled = segment_sampled.astype(np.float32) / 2147483648.0
            if segment.channels > 1:
                segment_sampled = segment_sampled.reshape(-1, segment.channels).mean(axis=1)

            # Resample to 16kHz
            if segment.frame_rate != 16000:
                segment_sampled = librosa.resample(segment_sampled, orig_sr=segment.frame_rate, target_sr=16000)

            input_features = self.whisper_processor(segment_sampled, sampling_rate=16000, return_tensors="pt").input_features.to(
                self.whisper_model.device)

            with torch.no_grad():
                encoder_outputs = self.whisper_model.encoder(input_features)
                encoder_hidden_states = encoder_outputs.last_hidden_state
                features = encoder_hidden_states.squeeze(0).mean(dim=0)
            return features.cpu()

        except Exception as e:
            print(f"Error processing {row['conv_id']} at {row['onset_msec']}-{row['offset_msec']}: {e}")
            return None

    def parse_context_string(self, context_str, limit=None):
        """Parse context string from format 'speakerA:segment1, speakerB:segment2, ...'"""
        if pd.isna(context_str) or not context_str:
            return []

        turns = [turn.strip() for turn in context_str.split(',')]

        if limit is not None and len(turns) > limit:
            if self.context_mode == "past":
                # keep most recent past segments
                turns = turns[-limit:]
            elif self.context_mode == "future":
                # keep closest future segments
                turns = turns[:limit]
        return turns

    def build_context(self, row):
        """Concatenate dialogue context to target text based on specified mode"""
        target_text = f"{row['speaker']}:{row['speech_original']}"
        window_size = None if self.context_window == "max" else self.context_window

        context = target_text
        context_tokens = self.tokenizer.tokenize(context)

        # Case 1: no context
        if self.context_mode == "none":
            return target_text

        # Case 2: past context only
        if self.context_mode == "past" and 'past_context' in row:
            past_turns = self.parse_context_string(row['past_context'], limit=window_size)

            for i in range(len(past_turns) - 1, -1, -1):
                past_turn = past_turns[i]
                current_context = past_turn + self.sep_token + context
                current_context_tokens = self.tokenizer.encode(current_context)

                if len(current_context_tokens) <= self.max_length:
                    context = current_context
                else:
                    break
        # Case 3: future context only
        elif self.context_mode == "future" and 'future_context' in row:
            future_turns = self.parse_context_string(row['future_context'], limit=window_size)

            for future_turn in future_turns:
                current_context = context + self.sep_token + future_turn
                current_context_tokens = self.tokenizer.encode(current_context)

                if len(current_context_tokens) <= self.max_length:
                    context = current_context
                else:
                    break
        # Case 4: both past and future context
        elif self.context_mode == "both" and 'past_context' in row and 'future_context' in row:
            past_turns = self.parse_context_string(row['past_context'], limit=window_size)
            future_turns = self.parse_context_string(row['future_context'], limit=window_size)

            # Combine past and future context
            past_idx, future_idx = 0, 0
            can_add_past = len(past_turns) > 0
            can_add_future = len(future_turns) > 0

            while can_add_past or can_add_future:
                if can_add_past:
                    past_turn = past_turns[len(past_turns) - 1 - past_idx]  # adding from the most recent to oldest segment
                    current_context = past_turn + self.sep_token + context
                    current_context_tokens = self.tokenizer.encode(current_context)

                    if len(current_context_tokens) <= self.max_length:
                        context = current_context
                        context_tokens = current_context_tokens
                        past_idx += 1
                        if past_idx >= len(past_turns):
                            can_add_past = False
                    else:
                        can_add_past = False

                if can_add_future:
                    future_turn = future_turns[future_idx]  # adding from closet segment to further
                    current_context = context + self.sep_token + future_turn
                    current_context_tokens = self.tokenizer.encode(current_context)

                    if len(current_context_tokens) <= self.max_length:
                        context = current_context
                        context_tokens = current_context_tokens
                        future_idx += 1
                        if future_idx >= len(future_turns):
                            can_add_future = False
                    else:
                        can_add_future = False

                if not can_add_past and not can_add_future:
                    break
        return context

    def prepare_linguistic_features(self):
        """Prepare handcrafted linguistic features for each given segment"""
        self.linguistic_features = []
        linguistic_columns = [col for col in self.df.columns if col.startswith('ling_')]

        print(f"Found {len(linguistic_columns)} linguistic features")

        features_matrix = np.vstack([
            np.array([row[col] for col in linguistic_columns])
            for _, row in self.df.iterrows()
        ])
        mean = np.nanmean(features_matrix, axis=0)
        std = np.nanstd(features_matrix, axis=0)
        std[std == 0] = 1.0

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting linguistic features"):
            features = torch.tensor([
                (row[col] - mean[i]) / std[i]
                for i, col in enumerate(linguistic_columns)
            ], dtype=torch.float)
            self.linguistic_features.append(features)

    def prepare_prosodic_features(self):
        """Prepare handcrafted prosodic features for each given segment"""
        self.prosodic_features = []
        prosodic_columns = [col for col in self.df.columns if col.startswith('pros_')]
        print(f"Found {len(prosodic_columns)} prosodic features")

        features_matrix = np.vstack([
            np.array([row[col] for col in prosodic_columns])
            for _, row in self.df.iterrows()
        ])

        mean = np.nanmean(features_matrix, axis=0)
        std = np.nanstd(features_matrix, axis=0)
        std[std == 0] = 1.0 

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting prosodic features"):
            features = torch.tensor([
                (row[col] - mean[i]) / std[i]
                for i, col in enumerate(prosodic_columns)
            ], dtype=torch.float)
            self.prosodic_features.append(features)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'conv_id': self.df.iloc[idx]['conv_id'],
            'onset_msec': self.df.iloc[idx]['onset_msec'],
            'offset_msec': self.df.iloc[idx]['offset_msec']
        }

        if self.use_text:
            item['input_ids'] = self.text_inputs[idx]['input_ids']
            item['attention_mask'] = self.text_inputs[idx]['attention_mask']

        if self.use_audio:
            item['audio_features'] = self.audio_features[idx]

        if self.use_linguistic:
            item['linguistic_features'] = self.linguistic_features[idx]

        if self.use_prosodic:
            item['prosodic_features'] = self.prosodic_features[idx]

        if self.labels[idx] != -1:
            try:
                label_value = int(self.labels[idx])
            except (ValueError, TypeError):
                label_map = {"RD": 0, "OIR": 1}
                label_value = label_map.get(self.labels[idx], -1)
            item['labels'] = torch.tensor(label_value, dtype=torch.long)
        return item
