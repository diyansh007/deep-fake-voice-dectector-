# Experimental Forensic Extensions

This branch adds an experimental research prototype around the existing detector. It is not independently validated and must not be described as proven, production-ready or patentable.

## Included mechanisms

1. **Decision-triggered adaptive re-analysis** in the forensic module: run a coarse pass over overlapping fixed-duration windows, rank windows with a transparent heuristic combining score ambiguity, neighboring score changes and basic signal-quality proxies, then re-analyze selected regions with shifted windows. The fixed-duration model input and configured feature pipeline are retained.
2. **Cross-window acoustic continuity review**: compare compact Log-Mel signatures and detector-score changes between neighboring coarse windows, then flag abrupt changes for human review. This cannot prove an edit or synthetic insertion; normal speech changes can trigger flags.
3. **Transformation-sensitivity audit**: compare the detector score after modest gain changes, low-level noise, a resampling round trip and band-limiting. This is a robustness/shortcut sensitivity proxy, not a watermark detector. It cannot establish watermark presence or prove that a model relies on one.

The overall score fusion, quality score and "Inconclusive" rule are heuristic starting points, not calibrated probabilities or validated operating thresholds. Thresholds should be selected on held-out speakers and datasets with false-positive/false-negative costs explicitly reported. Do not use results as the sole evidence for consequential decisions.

## Run it

Ensure dependencies and a trained checkpoint compatible with the existing model architecture and configured FEATURE_TYPE are available. From the repository root, run:

    python forensic_inference.py path/to/audio.wav --model models/best_model.pth --output report.json

Optional arguments include --device cpu, --coarse-hop 2.0, --decision-threshold 0.5, --priority-threshold 0.30 and --max-reanalysis-regions 3.

The script does not train a model. If a compatible checkpoint is missing, inference will not run until one is trained or supplied.

## Validation before making claims

- Add integration tests using a known checkpoint.
- Validate on held-out speakers and separate datasets; check dataset leakage and duplicate audio.
- Report accuracy, precision, recall, F1, ROC-AUC, EER, false-positive rate and false-negative rate.
- Evaluate by noise, codec, duration, language, recording device and synthesis method.
- Compare adaptive analysis with a fixed-window baseline and report runtime and extra-window counts.
- For a watermark-specific claim, use a dataset with documented watermark presence/absence and known watermarking procedures. The current perturbation audit alone is not evidence of watermark detection.
- Run a targeted patent search and have the RBU IPR Cell or a patent professional assess the specific technical mechanism and claims before filing.
