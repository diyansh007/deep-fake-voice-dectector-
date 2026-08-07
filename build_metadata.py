import argparse
from datasets.metadata_builder import get_or_build_metadata
from config import DATASET_CONFIG
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build standardized metadata for a supported dataset.")
    parser.add_argument("--dataset", type=str, default="asvspoof2019", choices=list(DATASET_CONFIG.keys()), help="Dataset to build metadata for.")
    parser.add_argument("--output", type=str, default=None, help="Output metadata CSV path. If omitted, default dataset metadata path is used.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force metadata rebuild even if cached metadata already exists.")
    args = parser.parse_args()

    metadata = get_or_build_metadata(args.dataset, force_rebuild=args.force_rebuild)
    if metadata.empty:
        raise RuntimeError("No metadata generated. Verify dataset directories and parser logic.")
    output_path = args.output if args.output else DATASET_CONFIG[args.dataset].get("metadata_path")
    if output_path is None:
        raise RuntimeError("No output metadata path provided or configured for dataset.")
    from datasets.metadata_builder import save_metadata
    save_metadata(metadata, output_path)
    print(f"Saved metadata to {output_path} with {len(metadata)} entries.")
