import os
import pandas as pd
from utils import label_to_int
from config import LABEL_MAPPING

SPLIT_MAP = {
    "training": "train",
    "validation": "dev",
    "testing": "eval",
}


def parse_fake_or_real_dataset(root_path: str, variants: dict) -> pd.DataFrame:
    records = []
    for generator_key, variant_subpath in variants.items():
        variant_root = os.path.join(root_path, variant_subpath)
        if not os.path.isdir(variant_root):
            continue
        for split_dir, split_name in SPLIT_MAP.items():
            split_root = os.path.join(variant_root, split_dir)
            if not os.path.isdir(split_root):
                continue
            for audio_type in ["real", "fake"]:
                class_root = os.path.join(split_root, audio_type)
                if not os.path.isdir(class_root):
                    continue
                for dirpath, _, filenames in os.walk(class_root):
                    for file_name in filenames:
                        if not file_name.lower().endswith((".wav", ".flac", ".mp3")):
                            continue
                        file_path = os.path.join(dirpath, file_name)
                        records.append({
                            "file_path": file_path,
                            "label": label_to_int(audio_type, LABEL_MAPPING),
                            "dataset_name": "fake_or_real",
                            "split": split_name,
                            "speaker_id": os.path.splitext(file_name)[0],
                            "audio_type": audio_type,
                            "generator": generator_key,
                            "sample_rate": None,
                            "duration": None,
                            "is_corrupted": False,
                            "source_format": os.path.splitext(file_name)[1].lstrip(".").lower(),
                        })
    return pd.DataFrame(records)
