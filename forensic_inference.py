"""Command-line entry point for experimental adaptive forensic analysis."""
import argparse
import json
import os

from config import MODEL_DIR
from forensics.adaptive_analysis import analyze_audio_file


def main():
    parser = argparse.ArgumentParser(
        description="Run experimental adaptive forensic analysis on an audio file."
    )
    parser.add_argument("audio_file", help="Path to the input audio file.")
    parser.add_argument(
        "--model",
        default=os.path.join(MODEL_DIR, "best_model.pth"),
        help="Trained model checkpoint (defaults to models/best_model.pth).",
    )
    parser.add_argument("--device", default=None, help="Optional torch device: cpu, cuda, or mps.")
    parser.add_argument("--coarse-hop", type=float, default=2.0, help="Coarse window hop in seconds.")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--priority-threshold", type=float, default=0.30)
    parser.add_argument("--max-reanalysis-regions", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = analyze_audio_file(
        args.audio_file,
        args.model,
        device_name=args.device,
        coarse_hop_seconds=args.coarse_hop,
        priority_threshold=args.priority_threshold,
        max_reanalysis_regions=args.max_reanalysis_regions,
        decision_threshold=args.decision_threshold,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"Saved experimental forensic report to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
