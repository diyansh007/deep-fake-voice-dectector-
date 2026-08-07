import os
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from preprocessing.audio_cleaning import clean_audio_file, EmptyAudioError
from features.feature_extraction import extract_log_mel, extract_mfcc, extract_log_mel_delta
from config import CACHE_DIR, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


class SpoofAudioDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        split: str = None,
        feature_type: str = "log_mel",
        cache_dir: str = CACHE_DIR,
        augment_fn=None,
    ):
        self.feature_type = feature_type
        self.cache_dir = cache_dir
        self.augment_fn = augment_fn
        self.metadata = metadata if split is None else metadata[metadata["split"] == split].reset_index(drop=True)
        if "is_corrupted" in self.metadata.columns:
            self.metadata = self.metadata[~self.metadata["is_corrupted"]].reset_index(drop=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate()

    def _validate(self):
        if self.metadata.empty:
            split_values = self.metadata["split"].unique() if "split" in self.metadata.columns else []
            raise ValueError(f"No metadata found for split: {split_values}")
        if "file_path" not in self.metadata.columns or "label" not in self.metadata.columns:
            raise ValueError("Metadata must contain file_path and label columns.")
        if self.feature_type not in ["log_mel", "log_mel_delta", "mfcc"]:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        max_attempts = min(len(self.metadata), 10)
        attempt = 0
        current_idx = idx
        while attempt < max_attempts:
            row = self.metadata.iloc[current_idx]
            file_path = row["file_path"]
            label = int(row["label"])
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            cache_path = os.path.join(self.cache_dir, f"{file_id}.wav")
            feature_cache_dir = os.path.join(self.cache_dir, self.feature_type)
            os.makedirs(feature_cache_dir, exist_ok=True)
            feature_cache_path = os.path.join(feature_cache_dir, f"{file_id}.npy")

            try:
                audio = None
                if cache_path and os.path.exists(cache_path):
                    try:
                        audio, sr = sf.read(cache_path, always_2d=False)
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)
                        audio = audio.astype(np.float32)
                        if audio.size == 0:
                            raise EmptyAudioError(f"Cached audio is empty: {cache_path}")
                    except Exception:
                        audio = None
                        try:
                            os.remove(cache_path)
                        except Exception:
                            pass
                if audio is None:
                    audio = clean_audio_file(file_path, cache_path=cache_path)

                if self.augment_fn is not None:
                    audio = self.augment_fn(audio, TARGET_SAMPLE_RATE)

                if self.augment_fn is None and os.path.exists(feature_cache_path):
                    features = np.load(feature_cache_path)
                else:
                    if self.feature_type == "mfcc":
                        features = extract_mfcc(audio, sample_rate=TARGET_SAMPLE_RATE)
                        features = np.expand_dims(features, axis=0)
                    elif self.feature_type == "log_mel_delta":
                        features = extract_log_mel_delta(audio, sample_rate=TARGET_SAMPLE_RATE)
                    else:
                        features = extract_log_mel(audio, sample_rate=TARGET_SAMPLE_RATE)
                        features = np.expand_dims(features, axis=0)
                    if self.augment_fn is None:
                        np.save(feature_cache_path, features)

                features = torch.from_numpy(features.astype(np.float32))
                return features, label
            except Exception as exc:
                logger.warning("Skipping invalid audio file %s: %s", file_path, exc)
                attempt += 1
                current_idx = (current_idx + 1) % len(self.metadata)
                continue

        raise RuntimeError(f"Unable to load a valid sample after {max_attempts} attempts starting from index {idx}.")
