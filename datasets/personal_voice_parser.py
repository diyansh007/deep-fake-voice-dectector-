import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED


def _collect_files(folder: str, exts: List[str]) -> List[str]:
    files = []
    if not os.path.isdir(folder):
        return files
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and any(path.lower().endswith(e) for e in exts):
            files.append(os.path.abspath(path))
    return files


def parse_personal_voice_dataset(root_path: str) -> pd.DataFrame:
    """
    Parse a personal voice dataset located at `root_path` which is expected to contain:
      - original/   -> human recordings (*.m4a)
      - ai/         -> AI generated (*.wav)

    Produces a dataframe with randomized train/dev/eval splits (70/15/15) using the
    fixed random seed from config.RANDOM_SEED.
    """
    original_dir = os.path.join(root_path, "original")
    ai_dir = os.path.join(root_path, "ai")

    orig_files = _collect_files(original_dir, [".m4a", ".wav", ".flac", ".mp3"])  # prefer .m4a
    ai_files = _collect_files(ai_dir, [".wav", ".flac", ".mp3"])  # ai expected as wav

    records = []
    all_items = []
    for p in orig_files:
        all_items.append({"file_path": p, "label": 0, "audio_type": "human"})
    for p in ai_files:
        all_items.append({"file_path": p, "label": 1, "audio_type": "ai"})

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # First split into train and temp (30% reserved for val+test)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["label"])
    # Split temp into val and test equally (15% each of full data)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"])

    def _row_rec(row, split_name):
        file_path = os.path.abspath(row["file_path"])
        fname = os.path.basename(file_path)
        return {
            "file_path": file_path,
            "label": int(row["label"]),
            "dataset_name": "personal_voice",
            "split": split_name,
            "speaker_id": os.path.splitext(fname)[0],
            "audio_type": row.get("audio_type", "human"),
            "generator": "personal_voice",
            "sample_rate": None,
            "duration": None,
            "is_corrupted": False,
            "source_format": os.path.splitext(fname)[1].lstrip(".").lower(),
        }

    records.extend([_row_rec(r, "train") for _, r in train_df.iterrows()])
    records.extend([_row_rec(r, "dev") for _, r in val_df.iterrows()])
    records.extend([_row_rec(r, "eval") for _, r in test_df.iterrows()])

    return pd.DataFrame(records)
