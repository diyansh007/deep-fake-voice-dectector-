import argparse
import glob
import os
import matplotlib.pyplot as plt
import librosa.display

from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel_delta
from config import TARGET_SAMPLE_RATE, FEATURE_CONFIG, PLOTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("audio")
parser.add_argument("--output-dir", default=PLOTS_DIR, help="Folder to save the output image")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
for existing in glob.glob(os.path.join(args.output_dir, "*.png")):
    os.remove(existing)

# Clean audio exactly like inference.py
audio = clean_audio_file(args.audio)

# Extract the 3-channel tensor used by your CNN
features = extract_log_mel_delta(audio)

fig = plt.figure(figsize=(16, 12))

# Waveform
ax1 = plt.subplot(4,1,1)
librosa.display.waveshow(audio, sr=TARGET_SAMPLE_RATE)
ax1.set_title("Waveform")

titles = [
    "Channel 1 - Log-Mel",
    "Channel 2 - Delta",
    "Channel 3 - Delta-Delta"
]

for i in range(3):
    ax = plt.subplot(4,1,i+2)

    img = librosa.display.specshow(
        features[i],
        sr=TARGET_SAMPLE_RATE,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    ax.set_title(titles[i])
    plt.colorbar(img, ax=ax)

plt.tight_layout()

output_path = os.path.join(args.output_dir, "model_input_visualization.png")
if os.path.exists(output_path):
    os.remove(output_path)
plt.savefig(output_path, dpi=300)

print(f"Saved {output_path}")

plt.show()
