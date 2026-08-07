import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

METADATA_PATHS = {
    "asvspoof2019": os.path.join(METADATA_DIR, "asvspoof2019_metadata.csv"),
    "fake_or_real": os.path.join(METADATA_DIR, "fake_or_real_metadata.csv"),
    "wavefake": os.path.join(METADATA_DIR, "wavefake_metadata.csv"),
    "common_voice_hi": os.path.join(METADATA_DIR, "common_voice_hi_metadata.csv"),
    "personal_voice": os.path.join(METADATA_DIR, "personal_voice_metadata.csv"),
    "combined": os.path.join(METADATA_DIR, "metadata.csv"),
}
METADATA_PATH = METADATA_PATHS["asvspoof2019"]

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

RANDOM_SEED = 42
TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4.0  # seconds
AUDIO_LENGTH = int(TARGET_SAMPLE_RATE * TARGET_DURATION)
FEATURE_TYPES = ["log_mel", "log_mel_delta", "mfcc"]

FEATURE_CONFIG = {
    "n_mels": 128,
    "n_mfcc": 40,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "f_min": 20,
    "f_max": 7600,
}

FEATURE_TYPE = "log_mel_delta"
DATA_LOADER_WORKERS = max(1, min(4, (os.cpu_count() or 1) - 1))

DATASET_CONFIG = {
    "asvspoof2019": {
        "name": "ASVspoof2019",
        "type": "protocol",
        "metadata_path": METADATA_PATHS["asvspoof2019"],
        "protocol_paths": {
            "LA": os.path.join(BASE_DIR, "Main data set", "ASVspoof 2019", "LA", "LA", "ASVspoof2019_LA_cm_protocols"),
            "PA": os.path.join(BASE_DIR, "Main data set", "ASVspoof 2019", "PA", "PA", "ASVspoof2019_PA_cm_protocols"),
        },
        "audio_paths": {
            "LA": os.path.join(BASE_DIR, "Main data set", "ASVspoof 2019", "LA", "LA"),
            "PA": os.path.join(BASE_DIR, "Main data set", "ASVspoof 2019", "PA", "PA"),
        },
    },
    "fake_or_real": {
        "name": "Fake-or-Real",
        "type": "folder",
        "metadata_path": METADATA_PATHS["fake_or_real"],
        "root_path": os.path.join(BASE_DIR, "Main data set", "Fake-or-Real (FoR)"),
        "variants": {
            "for-2seconds": os.path.join("for-2sec", "for-2seconds"),
            "for-norm": os.path.join("for-norm", "for-norm"),
            "for-original": os.path.join("for-original", "for-original"),
            "for-rerecorded": os.path.join("for-rerec", "for-rerecorded"),
        },
    },
    "wavefake": {
        "name": "WaveFake",
        "type": "folder",
        "metadata_path": METADATA_PATHS["wavefake"],
        "root_path": os.path.join(BASE_DIR, "Main data set", "WaveFake"),
        "generated_audio_path": os.path.join(BASE_DIR, "Main data set", "WaveFake", "generated_audio"),
    },
    "common_voice_hi": {
        "name": "Common Voice Hindi",
        "type": "common_voice_hi",
        "metadata_path": os.path.join(METADATA_DIR, "common_voice_hi_metadata.csv"),
        "root_path": os.path.join(BASE_DIR, "Main data set", "cv-hindi", "hi"),
    },
    "personal_voice": {
        "name": "Personal Voice",
        "type": "personal_voice",
        "metadata_path": os.path.join(METADATA_DIR, "personal_voice_metadata.csv"),
        "root_path": os.path.join(BASE_DIR, "Main data set", "training voice model"),
    },
}

LABEL_MAPPING = {
    "bonafide": 0,
    "genuine": 0,
    "real": 0,
    "human": 0,
    "spoof": 1,
    "fake": 1,
    "cloned": 1,
    "ai": 1,
    "generated": 1,
    "synthesized": 1,
}

AUGMENTATION_CONFIG = {
    "noise_factor": 0.005,
    "time_shift_max": 0.1,
    "pitch_shift_steps": 1,
    "time_mask_param": 30,
    "freq_mask_param": 15,
    "reverb_prob": 0.3,
    "time_mask_prob": 0.3,
    "freq_mask_prob": 0.3,
}

TRAINING_CONFIG = {
    "batch_size": 48,
    "learning_rate": 2e-5,
    "weight_decay": 3e-4,
    "epochs": 80,
    "patience": 10,
    "gradient_clip_val": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 1e-6,
    "early_stop_monitor": "f1",
}

# Streaming / real-time inference defaults
STREAM_WINDOW = 4.0  # seconds to buffer for a single decision
STREAM_HOP = 1.0     # seconds between decisions (how often to run inference)
STREAM_THRESHOLD = 0.5
STREAM_SMOOTHING_ALPHA = 0.6  # EMA alpha for smoothing ai probability
STREAM_DEVICE = None  # auto-detect device when loading model
