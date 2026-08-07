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
from config import TARGET_SAMPLE_RATE, MODEL_DIR


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
        features = extract_log_mel_delta(audio, sample_rate=TARGET_SAMPLE_RATE)
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


def run_inference(file_path: str, model_name: str = "best_model.pth", feature_type: str = "log_mel") -> dict:
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model, device = load_model(model_path)
    return infer_audio(model, device, file_path, feature_type=feature_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spoof detection inference on a single audio file or microphone input.")
    parser.add_argument("file_path", type=str, nargs="?", default=None, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Model filename in saved_models.")
    parser.add_argument("--microphone", action="store_true", help="Record from microphone for inference.")
    parser.add_argument("--duration", type=float, default=4.0, help="Microphone recording duration in seconds.")
    parser.add_argument("--feature", type=str, default="log_mel", choices=["log_mel", "mfcc", "log_mel_delta"], help="Feature type for inference.")
    args = parser.parse_args()

    if args.microphone:
        args.file_path = record_microphone_output(duration=args.duration)
        print(f"Recorded microphone audio to {args.file_path}")

    if not args.file_path:
        parser.error("Provide an audio file path or use --microphone.")

    result = run_inference(args.file_path, args.model, feature_type=args.feature)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Risk: {result['risk']}")
    print(f"Probabilities: {result['probabilities']}")
