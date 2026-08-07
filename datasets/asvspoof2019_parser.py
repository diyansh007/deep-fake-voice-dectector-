import os
import pandas as pd
from utils import label_to_int
from config import LABEL_MAPPING


def protocol_split_from_name(protocol_file: str) -> str:
    if "cm.train" in protocol_file:
        return "train"
    if "cm.dev" in protocol_file:
        return "dev"
    if "cm.eval" in protocol_file:
        return "eval"
    return "unknown"


def parse_asvspoof2019_protocol(protocol_path: str, audio_root: str, split_name: str) -> pd.DataFrame:
    records = []
    protocol_file = None
    for file_name in os.listdir(protocol_path):
        if file_name.endswith(".txt") and f"cm.{split_name}" in file_name:
            protocol_file = file_name
            break
    if protocol_file is None:
        return pd.DataFrame(records)
    protocol_file_path = os.path.join(protocol_path, protocol_file)
    with open(protocol_file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 5:
                continue
            speaker_id, file_id, _, _, label = fields[:5]
            file_name = f"{file_id}.flac"
            audio_path = os.path.join(audio_root, file_name)
            if not os.path.exists(audio_path):
                continue
            records.append({
                "file_path": audio_path,
                "label": label_to_int(label, LABEL_MAPPING),
                "dataset_name": "ASVspoof2019",
                "split": split_name,
                "speaker_id": speaker_id,
                "audio_type": label,
                "generator": "ASVspoof2019",
                "sample_rate": None,
                "duration": None,
                "is_corrupted": False,
                "source_format": "flac",
            })
    return pd.DataFrame(records)


def load_asvspoof2019_metadata() -> pd.DataFrame:
    frames = []
    for subset in ["LA", "PA"]:
        protocol_path = DATASET_CONFIG["asvspoof2019"]["protocol_paths"][subset]
        audio_root = os.path.join(DATASET_CONFIG["asvspoof2019"]["audio_paths"][subset], f"ASVspoof2019_{subset}_train", "flac")
        if os.path.isdir(protocol_path) and os.path.isdir(audio_root):
            subset_frame = parse_asvspoof2019_protocol(protocol_path, audio_root, split_name="train")
            frames.append(subset_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
