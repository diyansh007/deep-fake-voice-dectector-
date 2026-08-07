import os
from scipy.io import wavfile
from preprocessing.audio_cleaning import (
    load_audio,
    normalize_audio,
    trim_silence,
    pad_or_crop,
)

files = [
    "Main data set/training voice model/original/Recording (2).m4a",
    "Main data set/training voice model/original/Recording (5).m4a",
]

for f in files:
    audio, sr = load_audio(f)
    print("\n", f)
    print("Loaded     :", len(audio), "max=", float(audio.max()), "min=", float(audio.min()))

    norm = normalize_audio(audio)
    print("Normalized :", len(norm), "max=", float(norm.max()), "min=", float(norm.min()))

    trimmed = trim_silence(norm)
    print("Trimmed    :", len(trimmed), "max=", float(trimmed.max()), "min=", float(trimmed.min()))

    final = pad_or_crop(trimmed)
    print("Final      :", len(final), "max=", float(final.max()), "min=", float(final.min()))

    basename = os.path.splitext(os.path.basename(f))[0].replace(' ', '_')
    wavfile.write(f"{basename}_1_loaded.wav", sr, (norm * 32767).astype('int16'))
    wavfile.write(f"{basename}_2_trimmed.wav", sr, (trimmed * 32767).astype('int16'))
    wavfile.write(f"{basename}_3_final.wav", sr, (final * 32767).astype('int16'))
    print("Saved:", f"{basename}_1_loaded.wav", f"{basename}_2_trimmed.wav", f"{basename}_3_final.wav")
