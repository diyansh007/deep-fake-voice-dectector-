import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

from config import FEATURE_CONFIG
from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel_delta
from inference import load_model, infer_audio


parser = argparse.ArgumentParser()
parser.add_argument("audio")
args = parser.parse_args()

audio = clean_audio_file(args.audio)
features = extract_log_mel_delta(audio)
print("Tensor shape:", features.shape)

titles = [
    "Log-Mel Spectrogram",
    "Delta Spectrogram",
    "Delta-Delta Spectrogram",
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i in range(3):
    img = librosa.display.specshow(
        features[i],
        sr=16000,
        hop_length=FEATURE_CONFIG["hop_length"],
        x_axis="time",
        y_axis="mel",
        ax=axes[i],
    )
    axes[i].set_title(titles[i])
    fig.colorbar(img, ax=axes[i], format="%+2.0f dB")

plt.tight_layout()
output_path = os.path.join(args.output_dir, "spectrogram_channels.png")
if os.path.exists(output_path):
    os.remove(output_path)
plt.savefig(output_path, dpi=300)
print(f"Saved visualization to {output_path}")

model_path = os.path.join("models", "best_model_mixed.pth")
if os.path.exists(model_path):
    model, device = load_model(model_path)
    result = infer_audio(model, device, args.audio)
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print(f"Human      : {result['probabilities']['human'] * 100:.1f}%")
    print(f"AI         : {result['probabilities']['ai_generated'] * 100:.1f}%")
else:
    print(f"Model not found: {model_path}")

plt.show()
