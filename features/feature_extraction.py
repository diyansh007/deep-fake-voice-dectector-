import numpy as np
import librosa
from config import FEATURE_CONFIG


def extract_log_mel(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=FEATURE_CONFIG["n_fft"],
        hop_length=FEATURE_CONFIG["hop_length"],
        win_length=FEATURE_CONFIG["win_length"],
        n_mels=FEATURE_CONFIG["n_mels"],
        fmin=FEATURE_CONFIG["f_min"],
        fmax=FEATURE_CONFIG["f_max"],
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return log_mel.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=FEATURE_CONFIG["n_mfcc"],
        n_fft=FEATURE_CONFIG["n_fft"],
        hop_length=FEATURE_CONFIG["hop_length"],
        win_length=FEATURE_CONFIG["win_length"],
        fmin=FEATURE_CONFIG["f_min"],
        fmax=FEATURE_CONFIG["f_max"],
    )
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    return mfcc.astype(np.float32)


def extract_delta(features: np.ndarray, order: int = 1) -> np.ndarray:
    return librosa.feature.delta(features, order=order).astype(np.float32)


def extract_log_mel_delta(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    log_mel = extract_log_mel(audio, sample_rate)
    delta = extract_delta(log_mel, order=1)
    delta2 = extract_delta(log_mel, order=2)
    combined = np.stack([log_mel, delta, delta2], axis=0)
    return combined.astype(np.float32)


def extract_delta(features: np.ndarray, order: int = 1) -> np.ndarray:
    return librosa.feature.delta(features, order=order).astype(np.float32)


def extract_features(audio: np.ndarray, sample_rate: int = 16000, feature_types=None) -> dict:
    if feature_types is None:
        feature_types = ["log_mel", "mfcc", "delta_mfcc", "delta2_mfcc"]
    output = {}
    if "log_mel" in feature_types:
        output["log_mel"] = extract_log_mel(audio, sample_rate)
    if "mfcc" in feature_types:
        output["mfcc"] = extract_mfcc(audio, sample_rate)
    if "delta_mfcc" in feature_types:
        mfcc = output.get("mfcc") or extract_mfcc(audio, sample_rate)
        output["delta_mfcc"] = extract_delta(mfcc, order=1)
    if "delta2_mfcc" in feature_types:
        mfcc = output.get("mfcc") or extract_mfcc(audio, sample_rate)
        output["delta2_mfcc"] = extract_delta(mfcc, order=2)
    return output
