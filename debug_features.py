import numpy as np
import librosa

from preprocessing.audio_cleaning import clean_audio_file
from features.feature_extraction import extract_log_mel

a = clean_audio_file(
    "Main data set/training voice model/original/Recording (2).m4a"
)

b = clean_audio_file(
    "Main data set/training voice model/original/Recording (5).m4a"
)

print("Audio equal:", np.array_equal(a, b))

mel1 = librosa.feature.melspectrogram(y=a, sr=16000)
mel2 = librosa.feature.melspectrogram(y=b, sr=16000)

print("Raw mel equal:", np.array_equal(mel1, mel2))
print("Raw mel max diff:", np.max(np.abs(mel1 - mel2)))

feat1 = extract_log_mel(a)
feat2 = extract_log_mel(b)

print("Normalized equal:", np.array_equal(feat1, feat2))
print("Normalized max diff:", np.max(np.abs(feat1 - feat2)))
