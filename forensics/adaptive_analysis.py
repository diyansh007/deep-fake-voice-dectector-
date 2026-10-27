"""Experimental adaptive forensic analysis for the voice-spoof detector.

Adds shifted-window re-analysis, cross-window continuity review and a signal
transformation sensitivity audit. These are research prototypes, not validated
security decisions or proof of patentability.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import librosa
import torch

from config import AUDIO_LENGTH, FEATURE_TYPE, TARGET_SAMPLE_RATE
from features.feature_extraction import extract_log_mel, extract_log_mel_delta, extract_mfcc
from models.optimized_cnn import SpectrogramResNet
from preprocessing.audio_cleaning import load_audio, normalize_audio, pad_or_crop


def assess_audio_quality(audio: np.ndarray) -> dict[str, float]:
    """Compute transparent quality proxies; this is not an SNR estimator."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return {"rms": 0.0, "silence_fraction": 1.0,
                "clipping_fraction": 0.0, "quality_score": 0.0}
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    silence_fraction = float(np.mean(np.abs(x) < 0.005))
    clipping_fraction = float(np.mean(np.abs(x) >= 0.995))
    clip_penalty = min(1.0, clipping_fraction / 0.02)
    silence_penalty = min(1.0, max(0.0, (silence_fraction - 0.50) / 0.50))
    quality_score = float(np.clip(
        1.0 - 0.45 * clip_penalty - 0.55 * silence_penalty, 0.0, 1.0
    ))
    if rms < 1e-5:
        quality_score = 0.0
    return {"rms": rms, "silence_fraction": silence_fraction,
            "clipping_fraction": clipping_fraction, "quality_score": quality_score}


def window_starts(n_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    """Return window starts and include the tail-aligned final window."""
    if n_samples <= 0 or window_samples <= 0 or hop_samples <= 0:
        return []
    if n_samples <= window_samples:
        return [0]
    last_start = n_samples - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _fixed_window(audio: np.ndarray) -> np.ndarray:
    x = normalize_audio(np.asarray(audio, dtype=np.float32))
    return pad_or_crop(x, target_length=AUDIO_LENGTH).astype(np.float32)


def load_detector(model_path: str, device_name: str | None = None):
    """Load a checkpoint with the channel count expected by configured features."""
    if device_name:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    channels = 3 if FEATURE_TYPE == "log_mel_delta" else 1
    model = SpectrogramResNet(num_classes=2, in_channels=channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_ai_probability(model, device, audio: np.ndarray) -> float:
    """Infer one fixed-duration segment using the repository-configured features."""
    x = _fixed_window(audio)
    if FEATURE_TYPE == "log_mel_delta":
        features = extract_log_mel_delta(x, sample_rate=TARGET_SAMPLE_RATE)
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    elif FEATURE_TYPE == "mfcc":
        features = extract_mfcc(x, sample_rate=TARGET_SAMPLE_RATE)
        tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    else:
        features = extract_log_mel(x, sample_rate=TARGET_SAMPLE_RATE)
        tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return float(probabilities[1].item())


def _feature_signature(audio: np.ndarray) -> np.ndarray:
    """Compact spectral signature for relative continuity screening."""
    features = extract_log_mel(_fixed_window(audio), sample_rate=TARGET_SAMPLE_RATE)
    signature = features.mean(axis=1).astype(np.float64)
    norm = float(np.linalg.norm(signature))
    if norm > 1e-12:
        signature /= norm
    return signature


def analyze_continuity(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag abrupt spectral or score changes for review; never prove editing."""
    transitions = []
    for left, right in zip(windows, windows[1:]):
        a = np.asarray(left["_signature"], dtype=np.float64)
        b = np.asarray(right["_signature"], dtype=np.float64)
        cosine = float(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0
        ))
        spectral_distance = (1.0 - cosine) / 2.0
        probability_jump = abs(float(right["ai_probability"]) - float(left["ai_probability"]))
        score = float(0.65 * spectral_distance + 0.35 * probability_jump)
        transitions.append({
            "start_seconds": round(float(left["start_seconds"]), 3),
            "end_seconds": round(float(right["start_seconds"]), 3),
            "spectral_distance": round(float(spectral_distance), 4),
            "probability_jump": round(float(probability_jump), 4),
            "continuity_score": round(score, 4),
        })
    if not transitions:
        return []
    scores = np.asarray([row["continuity_score"] for row in transitions], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = max(0.40, median + 3.0 * 1.4826 * mad)
    for row in transitions:
        candidate = bool(row["continuity_score"] > threshold)
        row["review_threshold"] = round(float(threshold), 4)
        row["review_candidate"] = candidate
        row["interpretation"] = (
            "Abrupt transition worth review; normal speech changes can also trigger this flag."
            if candidate else "No transition exceeded this heuristic screening threshold."
        )
    return transitions


def _window_record(model, device, audio: np.ndarray, start: int,
                   window_samples: int) -> dict[str, Any]:
    segment = audio[start:min(start + window_samples, len(audio))]
    p_ai = predict_ai_probability(model, device, segment)
    return {
        "start_seconds": float(start / TARGET_SAMPLE_RATE),
        "end_seconds": float(min(start + window_samples, len(audio)) / TARGET_SAMPLE_RATE),
        "ai_probability": p_ai,
        "human_probability": float(1.0 - p_ai),
        "quality": assess_audio_quality(segment),
        "_audio": segment,
        "_signature": _feature_signature(segment),
    }


def _priority(records: list[dict[str, Any]]) -> list[float]:
    """Rank windows using uncertainty, neighboring score jumps and quality proxies."""
    values = []
    for i, record in enumerate(records):
        p = float(record["ai_probability"])
        uncertainty = 1.0 - abs(2.0 * p - 1.0)
        jumps = []
        if i > 0:
            jumps.append(abs(p - float(records[i - 1]["ai_probability"])))
        if i + 1 < len(records):
            jumps.append(abs(p - float(records[i + 1]["ai_probability"])))
        local_jump = max(jumps, default=0.0)
        quality = float(record["quality"]["quality_score"])
        values.append(float(
            0.50 * uncertainty + 0.30 * min(1.0, 2.0 * local_jump)
            + 0.20 * (1.0 - quality)
        ))
    return values


def _select_fine_starts(records: list[dict[str, Any]], priorities: list[float],
                        n_samples: int, window_samples: int,
                        initial_starts: set[int], max_regions: int,
                        priority_threshold: float) -> list[int]:
    if n_samples <= window_samples or max_regions <= 0:
        return []
    max_start = n_samples - window_samples
    chosen = []
    ordered = sorted(range(len(records)), key=lambda i: priorities[i], reverse=True)
    for index in ordered:
        if priorities[index] < priority_threshold:
            continue
        coarse_start = int(round(records[index]["start_seconds"] * TARGET_SAMPLE_RATE))
        # Same-duration shifted windows preserve the checkpoint's input shape.
        offsets = (-int(0.5 * TARGET_SAMPLE_RATE), int(0.5 * TARGET_SAMPLE_RATE),
                   int(1.0 * TARGET_SAMPLE_RATE))
        for offset in offsets:
            start = max(0, min(max_start, coarse_start + offset))
            if start not in initial_starts and start not in chosen:
                chosen.append(start)
                if len(chosen) >= max_regions * 3:
                    return chosen
    return chosen


def _transformation_sensitivity_audit(model, device, audio: np.ndarray) -> dict[str, Any]:
    """Compare scores under mild transforms; this is NOT a watermark detector."""
    x = _fixed_window(audio)
    rng = np.random.default_rng(42)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    noise_scale = max(rms, 0.01) * 0.002
    down = librosa.resample(x, orig_sr=TARGET_SAMPLE_RATE, target_sr=12000)
    roundtrip = librosa.resample(down, orig_sr=12000, target_sr=TARGET_SAMPLE_RATE)
    if len(roundtrip) < len(x):
        roundtrip = np.pad(roundtrip, (0, len(x) - len(roundtrip)))
    roundtrip = roundtrip[:len(x)]
    spectrum = np.fft.rfft(x)
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / TARGET_SAMPLE_RATE)
    spectrum[frequencies > 6000.0] = 0
    band_limited = np.fft.irfft(spectrum, n=len(x)).astype(np.float32)
    clipped = np.clip(x, -0.25, 0.25).astype(np.float32)
    variants = {
        "baseline": x,
        # Gain-only variants are intentionally omitted because the standard
        # preprocessing normalizes peak amplitude, largely cancelling pure gain.
        "low_level_noise": x + rng.normal(0.0, noise_scale, size=x.shape).astype(np.float32),
        "resample_roundtrip_12khz": roundtrip.astype(np.float32),
        "band_limited_6khz": band_limited,
        "clipped_amplitude": clipped,
    }
    scores = {name: predict_ai_probability(model, device, value) for name, value in variants.items()}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    spread = float(values.max() - values.min())
    return {
        "audit_type": "transformation_sensitivity_proxy",
        "scores": {name: round(float(score), 6) for name, score in scores.items()},
        "score_spread": round(spread, 6),
        "sensitivity_flag": bool(spread >= 0.20),
        "interpretation": (
            "The score changes substantially under a tested transformation. This is a robustness/shortcut warning only; "
            "it does not establish watermark presence or dependence."
            if spread >= 0.20 else
            "No large score spread under these transforms. This does not rule out watermark or shortcut dependence."
        ),
    }


def _public_window(record: dict[str, Any], priority: float | None = None,
                   stage: str = "coarse") -> dict[str, Any]:
    result = {
        "start_seconds": round(float(record["start_seconds"]), 3),
        "end_seconds": round(float(record["end_seconds"]), 3),
        "ai_probability": round(float(record["ai_probability"]), 6),
        "human_probability": round(float(record["human_probability"]), 6),
        "quality": {key: round(float(value), 6) for key, value in record["quality"].items()},
        "stage": stage,
    }
    if priority is not None:
        result["reanalysis_priority"] = round(float(priority), 6)
    return result


def analyze_audio_file(audio_path: str, model_path: str, device_name: str | None = None,
                       coarse_hop_seconds: float = 2.0, priority_threshold: float = 0.30,
                       max_reanalysis_regions: int = 3,
                       decision_threshold: float = 0.50) -> dict[str, Any]:
    """Run coarse scan, selected re-analysis, continuity review and score-sensitivity audit."""
    if coarse_hop_seconds <= 0:
        raise ValueError("coarse_hop_seconds must be positive")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1")
    if max_reanalysis_regions < 0:
        raise ValueError("max_reanalysis_regions cannot be negative")

    audio, _ = load_audio(audio_path, sample_rate=TARGET_SAMPLE_RATE)
    audio = normalize_audio(audio.astype(np.float32))
    if audio.size == 0:
        raise ValueError("Audio file has no samples after decoding.")
    model, device = load_detector(model_path, device_name=device_name)
    window_samples = AUDIO_LENGTH
    hop_samples = max(1, int(coarse_hop_seconds * TARGET_SAMPLE_RATE))
    starts = window_starts(len(audio), window_samples, hop_samples)
    coarse = [_window_record(model, device, audio, start, window_samples) for start in starts]
    priorities = _priority(coarse)
    fine_starts = _select_fine_starts(
        coarse, priorities, len(audio), window_samples, set(starts),
        max_reanalysis_regions, priority_threshold
    )
    fine = [_window_record(model, device, audio, start, window_samples) for start in fine_starts]
    continuity = analyze_continuity(coarse)
    review_index = int(np.argmax(priorities)) if priorities else 0
    sensitivity = _transformation_sensitivity_audit(model, device, coarse[review_index]["_audio"])

    weighted_scores = []
    for record in coarse:
        quality = float(record["quality"]["quality_score"])
        weighted_scores.append((float(record["ai_probability"]), max(0.10, quality)))
    for record in fine:
        quality = float(record["quality"]["quality_score"])
        weighted_scores.append((float(record["ai_probability"]), 0.35 * max(0.10, quality)))
    total_weight = sum(weight for _, weight in weighted_scores)
    final_score = float(sum(score * weight for score, weight in weighted_scores) / max(total_weight, 1e-12))
    mean_quality = float(np.mean([item["quality"]["quality_score"] for item in coarse]))
    score_std = float(np.std([item["ai_probability"] for item in coarse]))
    if mean_quality < 0.35 or abs(final_score - decision_threshold) <= 0.05 or score_std >= 0.30:
        decision = "Inconclusive"
    else:
        decision = "AI-generated" if final_score >= decision_threshold else "Human"

    return {
        "audio_file": str(audio_path),
        "model_checkpoint": str(model_path),
        "feature_type": FEATURE_TYPE,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "window_duration_seconds": window_samples / TARGET_SAMPLE_RATE,
        "coarse_hop_seconds": coarse_hop_seconds,
        "decision_threshold": decision_threshold,
        "decision": decision,
        "ai_probability_heuristic_fusion": round(final_score, 6),
        "mean_audio_quality_proxy": round(mean_quality, 6),
        "coarse_score_std": round(score_std, 6),
        "decision_note": (
            "Fusion and inconclusive rules are experimental heuristics, not calibrated probabilities. "
            "Validate thresholds on held-out data before operational use."
        ),
        "coarse_windows": [
            _public_window(record, priorities[i], "coarse") for i, record in enumerate(coarse)
        ],
        "reanalysis_windows": [_public_window(record, stage="adaptive_reanalysis") for record in fine],
        "continuity_review": continuity,
        "transformation_sensitivity_audit": sensitivity,
        "feature_limitations": [
            "Transformation sensitivity is not a watermark detector and cannot prove watermark presence.",
            "Continuity flags mark regions worth reviewing; natural speech changes can also trigger them.",
            "This prototype makes no independent accuracy or patentability claim.",
        ],
    }
