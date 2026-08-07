import os
import pandas as pd

def parse_common_voice_hi_dataset(root_path: str) -> pd.DataFrame:
    records = []
    for split_name in ["train", "dev"]:
        tsv_path = os.path.join(root_path, f"{split_name}.tsv")
        clips_dir = os.path.join(root_path, "clips")
        if not os.path.isfile(tsv_path) or not os.path.isdir(clips_dir):
            continue

        try:
            df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
        except Exception:
            continue

        if "path" not in df.columns:
            continue

        for _, row in df.iterrows():
            rel_path = str(row["path"]).strip()
            if not rel_path:
                continue
            audio_path = os.path.abspath(os.path.join(clips_dir, rel_path))
            if not os.path.exists(audio_path):
                continue

            records.append({
                "file_path": audio_path,
                "label": 0,
                "dataset_name": "common_voice_hi",
                "split": split_name,
                "speaker_id": str(row.get("client_id", os.path.splitext(os.path.basename(rel_path))[0])),
                "audio_type": "human",
                "generator": "common_voice_hi",
                "sample_rate": None,
                "duration": None,
                "is_corrupted": False,
                "source_format": os.path.splitext(audio_path)[1].lstrip(".").lower(),
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
