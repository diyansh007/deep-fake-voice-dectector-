import argparse
import glob
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import torch

from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel_delta
from inference import load_model
from config import TARGET_SAMPLE_RATE, PLOTS_DIR


def visualize_audio(audio, features, feature_maps, probs, prediction, output_path):
    fig = plt.figure(figsize=(18, 18))

    # Waveform
    ax1 = fig.add_subplot(5, 4, 1)
    librosa.display.waveshow(audio, sr=TARGET_SAMPLE_RATE, ax=ax1)
    ax1.set_title("Waveform")
    ax1.set_xlabel("")

    # Log-Mel / Delta / Delta-Delta
    titles = ["Log-Mel", "Delta", "Delta-Delta"]
    for i in range(3):
        ax = fig.add_subplot(5, 4, 2 + i)
        img = librosa.display.specshow(
            features[i],
            sr=TARGET_SAMPLE_RATE,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(titles[i])
        fig.colorbar(img, ax=ax, format="%.2f")

    # CNN feature maps: first 16 maps from stem output
    num_maps = min(16, feature_maps.shape[0])
    for i in range(num_maps):
        ax = fig.add_subplot(5, 4, 5 + i)
        ax.imshow(feature_maps[i].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Feature Map {i+1}")
        ax.axis("off")

    # Prediction / probabilities box
    human_prob = float(probs[0])
    ai_prob = float(probs[1])
    confidence = float(probs.max())
    predicted_label = "Human" if prediction == 0 else "AI-generated"

    stats = (
        f"Prediction : {predicted_label}\n"
        f"Confidence : {confidence*100:.2f}%\n"
        f"Human      : {human_prob:.3f}\n"
        f"AI         : {ai_prob:.3f}\n"
    )

    fig.text(0.02, 0.02, stats, fontsize=16, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model input, feature maps, and prediction.")
    parser.add_argument("audio", help="Path to an audio file to visualize")
    parser.add_argument("--model", default="best_model_mixed.pth", help="Model file name in models/")
    parser.add_argument("--output-dir", default=PLOTS_DIR, help="Folder to save the output image")
    parser.add_argument("--output", default="cnn_dashboard.png", help="Output image name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for existing in glob.glob(os.path.join(args.output_dir, "*.png")):
        os.remove(existing)

    model, device = load_model("models/" + args.model)
    model.eval()

    audio = clean_audio_file(args.audio)
    features = extract_log_mel_delta(audio, sample_rate=TARGET_SAMPLE_RATE)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    activation = {}

    def hook(module, input, output):
        activation["stem"] = output.detach().cpu()[0]

    handle = model.stem.register_forward_hook(hook)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu()
        prediction = int(torch.argmax(probs).item())

    handle.remove()

    if "stem" not in activation:
        raise RuntimeError("Failed to capture stem activations")

    feature_maps = activation["stem"]
    output_path = os.path.join(args.output_dir, args.output)
    if os.path.exists(output_path):
        os.remove(output_path)
    visualize_audio(audio, features, feature_maps, probs, prediction, output_path)
