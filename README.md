# Deep Fake Voice Detector

A production-style PyTorch system for spoof / scam voice detection.

## Project Overview

This project supports:
- ASVspoof 2019 as the first dataset
- reusable metadata parsing for future datasets (ASVspoof 2021, WaveFake, etc.)
- a standardized dataset format with unified labels
- reusable audio cleaning and feature extraction pipelines
- a CNN classifier for spectrogram-based spoof detection
- inference for WAV and future audio file support

## Structure

- `config.py` - global configuration and dataset paths
- `datasets/` - metadata builders and dataset parsers
- `preprocessing/` - audio cleaning and augmentation
- `features/` - feature extraction logic
- `models/` - model definitions
- `train.py` - training workflow
- `evaluate.py` - evaluation workflow
- `inference.py` - inference for single files
- `utils.py` - metrics, logging, and helpers
- `saved_models/` - model checkpoints
- `cache/` - audio cache
- `plots/` - training and evaluation plots

## Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure ASVspoof 2019 data is available under `data set/archive/LA/LA` and `data set/archive/PA/PA`.

3. Train the model:

```bash
python train.py
```

4. Evaluate the saved model:

```bash
python evaluate.py
```

5. Run inference:

```bash
python inference.py path/to/audio.wav
```

## Standardized Metadata Format

Metadata CSV columns:
- `file_path`
- `label`
- `dataset_name`
- `split`
- `speaker_id`
- `audio_type`
- `sample_rate`
- `duration`
- `is_corrupted`
- `source_format`

Labels are mapped consistently:
- `bonafide`, `genuine`, `real`, `human` -> `0`
- `spoof`, `fake`, `cloned`, `ai` -> `1`

## Notes

- The feature pipeline uses Log-Mel Spectrogram as the main input.
- Augmentation is applied only during training.
- The model and pipeline are designed to support future datasets without rewriting core training logic.


## Experimental Forensic Analysis (research prototype)

A separate experimental entry point adds decision-triggered shifted-window re-analysis, cross-window acoustic-continuity review, and a transformation-sensitivity audit. It leaves the original offline and streaming entry points unchanged while the additions are validated.

    python forensic_inference.py path/to/audio.wav --model models/best_model.pth --output report.json

Read docs/EXPERIMENTAL_FORENSICS.md before interpreting results. Adaptive fusion and quality metrics are heuristics; continuity flags are review prompts; the transformation audit is not a watermark detector and cannot prove watermark dependence. These extensions are research prototypes, not validated or demonstrated to be patentable.
