import random
import numpy as np
import librosa
from config import AUGMENTATION_CONFIG, TARGET_SAMPLE_RATE


def add_random_noise(audio: np.ndarray, noise_factor: float = AUGMENTATION_CONFIG["noise_factor"]) -> np.ndarray:
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def random_gain(audio: np.ndarray, min_gain: float = 0.8, max_gain: float = 1.2) -> np.ndarray:
    gain = random.uniform(min_gain, max_gain)
    return audio * gain


def time_shift(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, max_shift: float = AUGMENTATION_CONFIG["time_shift_max"]) -> np.ndarray:
    shift = int(random.uniform(-max_shift, max_shift) * sample_rate)
    return np.roll(audio, shift)


def time_stretch(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, min_rate: float = 0.9, max_rate: float = 1.1) -> np.ndarray:
    try:
        rate = random.uniform(min_rate, max_rate)
        stretched = librosa.effects.time_stretch(audio, rate)
        if len(stretched) > len(audio):
            stretched = stretched[: len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode="constant")
        return stretched
    except Exception:
        return audio


def pitch_shift(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, n_steps: int = AUGMENTATION_CONFIG["pitch_shift_steps"]) -> np.ndarray:
    try:
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=random.uniform(-n_steps, n_steps))
    except Exception:
        return audio


def time_mask(audio: np.ndarray, max_mask_pct: int = AUGMENTATION_CONFIG["time_mask_param"]) -> np.ndarray:
    mask_len = int(random.uniform(0, max_mask_pct))
    if mask_len <= 0:
        return audio
    start = random.randint(0, max(0, len(audio) - mask_len))
    audio[start:start + mask_len] = 0
    return audio


def freq_mask(spectrogram: np.ndarray, max_mask_pct: int = AUGMENTATION_CONFIG["freq_mask_param"]) -> np.ndarray:
    num_mel_channels = spectrogram.shape[0]
    mask_len = int(random.uniform(0, max_mask_pct))
    start = random.randint(0, max(0, num_mel_channels - mask_len))
    spectrogram[start:start + mask_len, :] = 0
    return spectrogram


def light_reverb(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, reverb_prob: float = AUGMENTATION_CONFIG["reverb_prob"]) -> np.ndarray:
    if random.random() > reverb_prob:
        return audio
    decay = random.uniform(0.1, 0.3)
    impulse = np.zeros(int(sample_rate * 0.03))
    impulse[0] = 1.0
    for i in range(1, len(impulse)):
        impulse[i] = impulse[i-1] * (1 - decay)
    return np.convolve(audio, impulse, mode="same")


def augment_audio(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio = random_gain(audio)
    audio = add_random_noise(audio)
    audio = time_shift(audio, sample_rate)
    if random.random() < 0.5:
        audio = time_stretch(audio, sample_rate)
    audio = pitch_shift(audio, sample_rate)
    audio = light_reverb(audio, sample_rate)
    return audio
