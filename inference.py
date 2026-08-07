import glob
import os
import argparse
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel, extract_mfcc, extract_log_mel_delta
from models.optimized_cnn import SpectrogramResNet
from utils import int_to_label, risk_from_probability
from config import TARGET_SAMPLE_RATE, MODEL_DIR, FEATURE_TYPE, PLOTS_DIR


def load_model(model_path: str, device: str = None):
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    model = SpectrogramResNet(num_classes=2, in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def infer_audio(model, device, file_path: str, feature_type: str = "log_mel"):
    audio = clean_audio_file(file_path)
    if feature_type == "mfcc":
        features = extract_mfcc(audio, sample_rate=TARGET_SAMPLE_RATE)
        tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    elif feature_type == "log_mel_delta":
        features = extract_log_mel_delta(
            audio,
            sample_rate=TARGET_SAMPLE_RATE
        )
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    else:
        features = extract_log_mel(audio, sample_rate=TARGET_SAMPLE_RATE)
        tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))
    return {
        "file_path": file_path,
        "prediction": int_to_label(prediction),
        "confidence": float(np.max(probabilities)),
        "risk": risk_from_probability(float(np.max(probabilities))),
        "probabilities": {
            "human": float(probabilities[0]),
            "ai_generated": float(probabilities[1]),
        },
    }


def record_microphone_output(duration: float = 4.0, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    output_path = os.path.join(MODEL_DIR, "mic_input.wav")
    sf.write(output_path, audio.squeeze(), sample_rate)
    return output_path


def clear_plot_outputs(output_dir: str):
    for existing in glob.glob(os.path.join(output_dir, "*.png")):
        os.remove(existing)


def save_visualization(audio, model, device, file_path: str, output_dir: str):
    from features.feature_extraction import extract_log_mel_delta

    os.makedirs(output_dir, exist_ok=True)
    clear_plot_outputs(output_dir)

    audio = clean_audio_file(file_path)
    features = extract_log_mel_delta(audio, sample_rate=TARGET_SAMPLE_RATE)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    activation = {}

    def hook(module, input, output):
        activation["stem"] = output.detach().cpu()[0]

    handle = model.stem.register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    handle.remove()

    if "stem" not in activation:
        raise RuntimeError("Failed to capture stem activations")

    feature_maps = activation["stem"]
    output_path = os.path.join(output_dir, "cnn_dashboard.png")
    if os.path.exists(output_path):
        os.remove(output_path)

    import matplotlib.pyplot as plt
    import librosa.display

    fig = plt.figure(figsize=(18, 18))
    ax1 = fig.add_subplot(5, 4, 1)
    librosa.display.waveshow(audio, sr=TARGET_SAMPLE_RATE, ax=ax1)
    ax1.set_title("Waveform")
    ax1.set_xlabel("")

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

    num_maps = min(16, feature_maps.shape[0])
    for i in range(num_maps):
        ax = fig.add_subplot(5, 4, 5 + i)
        ax.imshow(feature_maps[i].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Feature Map {i+1}")
        ax.axis("off")

    probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    prediction = int(torch.argmax(probs).item())
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
    print(f"Saved visualization to {output_path}")


def run_inference(file_path: str, model_name: str = "best_model.pth", feature_type: str = FEATURE_TYPE) -> dict:
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model, device = load_model(model_path)
    save_visualization(audio=None, model=model, device=device, file_path=file_path, output_dir=PLOTS_DIR)
    return infer_audio(model, device, file_path, feature_type=feature_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run spoof detection inference."
    )

    parser.add_argument("file_path", nargs="?", default=None)
    parser.add_argument("--model", default="best_model.pth")
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--duration", type=float, default=4.0)

    parser.add_argument(
        "--feature",
        default=FEATURE_TYPE,
        choices=["log_mel", "mfcc", "log_mel_delta"],
    )

    args = parser.parse_args()

    if args.microphone:
        args.file_path = record_microphone_output(args.duration)

    if not args.file_path:
        parser.error("Provide an audio file.")

    result = run_inference(
        args.file_path,
        args.model,
        feature_type=args.feature,
    )

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Risk: {result['risk']}")
    print(result["probabilities"])
