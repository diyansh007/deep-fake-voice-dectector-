import os
import soundfile as sf
import librosa
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional
from config import TARGET_SAMPLE_RATE, AUDIO_LENGTH


class AudioCleaningError(Exception):
    pass


class EmptyAudioError(AudioCleaningError):
    pass


def load_audio(file_path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    try:
        data, sr = sf.read(file_path, always_2d=False)
    except Exception as first_exc:
        try:
            data, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception as second_exc:
            try:
                waveform, sr = torchaudio.load(file_path)
                waveform = waveform.mean(dim=0).numpy()
                data = waveform.astype(np.float32)
            except Exception as third_exc:
                raise ValueError(
                    f"Could not decode audio: {file_path}. "
                    f"soundfile error: {first_exc}; librosa error: {second_exc}; torchaudio error: {third_exc}"
                )
    if data is None:
        raise EmptyAudioError(f"Could not decode audio: {file_path}")
    if isinstance(data, np.ndarray) and data.size == 0:
        raise EmptyAudioError(f"Audio data is empty: {file_path}")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != sample_rate:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    data = data.astype(np.float32)
    if data.size == 0:
        raise EmptyAudioError(f"Audio is empty after resampling: {file_path}")
    return data, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio is None or audio.size == 0:
        raise EmptyAudioError("Empty audio cannot be normalized.")
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def pad_or_crop(audio: np.ndarray, target_length: int = AUDIO_LENGTH) -> np.ndarray:
    if len(audio) > target_length:
        start = 0
        return audio[start:start + target_length]
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio


def clean_audio_file(file_path: str, cache_path: Optional[str] = None) -> np.ndarray:
    audio, sr = load_audio(file_path)
    if audio is None or audio.size == 0:
        raise EmptyAudioError(f"Empty audio after loading: {file_path}")
    audio = normalize_audio(audio)
    audio = trim_silence(audio)
    if audio is None or audio.size == 0:
        raise EmptyAudioError(f"Empty audio after trimming: {file_path}")
    audio = pad_or_crop(audio)
    if audio is None or audio.size == 0:
        raise EmptyAudioError(f"Empty audio after padding/cropping: {file_path}")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        sf.write(cache_path, audio, sr)
    return audio
