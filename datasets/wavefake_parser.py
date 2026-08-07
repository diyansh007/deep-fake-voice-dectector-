import os
import pandas as pd
from utils import label_to_int
from config import LABEL_MAPPING


def parse_wavefake_dataset(generator_root: str) -> pd.DataFrame:
    records = []
    if not os.path.isdir(generator_root):
        return pd.DataFrame(records)
    for dirpath, _, filenames in os.walk(generator_root):
        for file_name in filenames:
            if not file_name.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            file_path = os.path.join(dirpath, file_name)
            records.append({
                "file_path": file_path,
                "label": label_to_int("fake", LABEL_MAPPING),
                "dataset_name": "wavefake",
                "split": "test",
                "speaker_id": os.path.splitext(file_name)[0],
                "audio_type": "fake",
                "generator": os.path.basename(generator_root),
                "sample_rate": None,
                "duration": None,
                "is_corrupted": False,
                "source_format": os.path.splitext(file_name)[1].lstrip(".").lower(),
            })
    return pd.DataFrame(records)
