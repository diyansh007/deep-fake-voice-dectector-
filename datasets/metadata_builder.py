import os
import pandas as pd
from typing import Optional
from config import DATASET_CONFIG, LABEL_MAPPING
from preprocessing.audio_cleaning import load_audio, EmptyAudioError as AudioCleaningEmptyError
from datasets.asvspoof2019_parser import parse_asvspoof2019_protocol
from datasets.fake_or_real_parser import parse_fake_or_real_dataset
from datasets.wavefake_parser import parse_wavefake_dataset
from datasets.common_voice_hi_parser import parse_common_voice_hi_dataset
from datasets.personal_voice_parser import parse_personal_voice_dataset


def build_metadata_for_asvspoof2019() -> pd.DataFrame:
    records = []
    dataset_cfg = DATASET_CONFIG["asvspoof2019"]
    for subset in ["LA", "PA"]:
        protocol_path = dataset_cfg["protocol_paths"][subset]
        audio_base = dataset_cfg["audio_paths"][subset]
        for split in ["train", "dev", "eval"]:
            data_dir = os.path.join(audio_base, f"ASVspoof2019_{subset}_{split}", "flac")
            if not os.path.isdir(data_dir):
                continue
            subset_records = parse_asvspoof2019_protocol(protocol_path, data_dir, split_name=split)
            if not subset_records.empty:
                records.append(subset_records)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df = validate_metadata(df)
    return df


def build_metadata_for_fake_or_real(force_rebuild: bool = False) -> pd.DataFrame:
    dataset_cfg = DATASET_CONFIG["fake_or_real"]
    root_path = dataset_cfg["root_path"]
    if not os.path.isdir(root_path):
        return pd.DataFrame()
    records = parse_fake_or_real_dataset(root_path, dataset_cfg["variants"])
    if records.empty:
        return pd.DataFrame()
    return validate_metadata(records)


def build_metadata_for_wavefake(force_rebuild: bool = False) -> pd.DataFrame:
    dataset_cfg = DATASET_CONFIG["wavefake"]
    generated_audio_path = dataset_cfg["generated_audio_path"]
    if not os.path.isdir(generated_audio_path):
        return pd.DataFrame()
    records = parse_wavefake_dataset(generated_audio_path)
    if records.empty:
        return pd.DataFrame()
    return validate_metadata(records)


def build_metadata_for_common_voice_hi() -> pd.DataFrame:
    dataset_cfg = DATASET_CONFIG["common_voice_hi"]
    root_path = dataset_cfg["root_path"]
    if not os.path.isdir(root_path):
        return pd.DataFrame()
    records = parse_common_voice_hi_dataset(root_path)
    if records.empty:
        return pd.DataFrame()
    return validate_metadata(records)


def get_or_build_metadata(dataset_name: str = "asvspoof2019", force_rebuild: bool = False) -> pd.DataFrame:
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_cfg = DATASET_CONFIG[dataset_name]
    metadata_path = dataset_cfg.get("metadata_path")
    if metadata_path and os.path.exists(metadata_path) and not force_rebuild:
        metadata = load_metadata(metadata_path)
        if not metadata.empty:
            return validate_metadata(metadata)

    if dataset_name == "asvspoof2019":
        metadata = build_metadata_for_asvspoof2019()
    elif dataset_name == "fake_or_real":
        metadata = build_metadata_for_fake_or_real(force_rebuild=force_rebuild)
    elif dataset_name == "wavefake":
        metadata = build_metadata_for_wavefake(force_rebuild=force_rebuild)
    elif dataset_name == "common_voice_hi":
        metadata = build_metadata_for_common_voice_hi()
    elif dataset_name == "personal_voice":
        dataset_cfg = DATASET_CONFIG["personal_voice"]
        root = dataset_cfg.get("root_path")
        if root and os.path.isdir(root):
            metadata = parse_personal_voice_dataset(root)
        else:
            metadata = pd.DataFrame()
    else:
        metadata = pd.DataFrame()

    if metadata.empty:
        raise RuntimeError(f"Unable to build {dataset_name} metadata: no records found.")

    if metadata_path:
        save_metadata(metadata, metadata_path)
    return metadata


def validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return metadata
    required_columns = ["file_path", "label", "dataset_name", "split", "speaker_id", "audio_type", "source_format", "generator", "is_corrupted"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata missing required columns: {missing_columns}")
    metadata = metadata.drop_duplicates(subset=["file_path"])
    metadata["file_path"] = metadata["file_path"].astype(str)
    metadata["dataset_name"] = metadata["dataset_name"].astype(str)
    metadata["split"] = metadata["split"].astype(str)
    metadata["speaker_id"] = metadata["speaker_id"].astype(str)
    metadata["audio_type"] = metadata["audio_type"].astype(str)
    metadata["generator"] = metadata["generator"].astype(str)
    metadata["source_format"] = metadata["source_format"].astype(str)
    metadata["is_corrupted"] = metadata["is_corrupted"].astype(bool)

    def get_validation_issue(path: str) -> Optional[str]:
        if not os.path.exists(path):
            return "missing file"
        try:
            if os.path.getsize(path) == 0:
                return "zero-byte file"
        except Exception:
            return "unreadable file"
        try:
            audio, sr = load_audio(path)
            if audio is None or len(audio) == 0:
                return "empty audio"
        except AudioCleaningEmptyError:
            return "empty audio"
        except Exception:
            return "unreadable or invalid audio"
        return None

    metadata["validation_issue"] = metadata["file_path"].apply(get_validation_issue)
    invalid_entries = metadata[metadata["validation_issue"].notna()].copy()
    if not invalid_entries.empty:
        total_files = len(metadata)
        removed_files = len(invalid_entries)
        valid_files = total_files - removed_files
        print("Metadata validation summary:")
        print(f"  Total files: {total_files}")
        print(f"  Valid files: {valid_files}")
        print(f"  Removed files: {removed_files}")
        for reason, count in invalid_entries["validation_issue"].value_counts().items():
            print(f"    {reason}: {count}")
        for bad_file in invalid_entries["file_path"]:
            print(f"    removed: {bad_file}")
    metadata.loc[metadata["validation_issue"].notna(), "is_corrupted"] = True
    metadata = metadata[metadata["is_corrupted"] == False].copy()
    metadata = metadata.drop(columns=["validation_issue"])
    metadata["sample_rate"] = metadata.get("sample_rate", None)
    metadata["duration"] = metadata.get("duration", None)
    return metadata


def save_metadata(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def load_metadata(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file does not exist: {csv_path}")
    return pd.read_csv(csv_path)
