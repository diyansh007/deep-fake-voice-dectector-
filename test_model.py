import torch
from inference import load_model
from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel_delta

model, device = load_model("models/best_model_personal_1.pth")

audio = clean_audio_file(
    "Main data set/training voice model/original/Recording (2).m4a"
)

features = extract_log_mel_delta(audio)

tensor = torch.from_numpy(features).unsqueeze(0).to(device)

print("Tensor shape:", tensor.shape)

with torch.no_grad():
    outputs = model(tensor)

print("Output:", outputs)
