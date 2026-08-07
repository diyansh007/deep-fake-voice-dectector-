import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix
from config import MODEL_DIR, PLOTS_DIR, TRAINING_CONFIG, FEATURE_TYPE, DATA_LOADER_WORKERS, DATASET_CONFIG
from utils import setup_logger, compute_metrics, compute_eer
from datasets.metadata_builder import get_or_build_metadata
from datasets.dataset import SpoofAudioDataset
from models.optimized_cnn import SpectrogramResNet
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model_path: str, metadata_path: str = None, split: str = "dev", dataset_name: str = "asvspoof2019"):
    logger = setup_logger("evaluate", os.path.join(MODEL_DIR, "evaluate.log"))
    if metadata_path is None or not os.path.exists(metadata_path):
        metadata = get_or_build_metadata(dataset_name)
    else:
        metadata = pd.read_csv(metadata_path)
    dataset = SpoofAudioDataset(metadata, split=split, feature_type=FEATURE_TYPE, augment_fn=None)
    loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=DATA_LOADER_WORKERS > 0,
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    in_channels = 3 if FEATURE_TYPE == "log_mel_delta" else 1
    model = SpectrogramResNet(num_classes=2, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_prob=torch.tensor(y_prob))
    metrics["eer"] = compute_eer(y_true, [p[1] for p in y_prob])
    logger.info(f"Evaluation metrics: {metrics}")
    plot_roc_curve(y_true, [p[1] for p in y_prob])
    plot_confusion_matrix(y_true, y_pred)
    return metrics


def plot_roc_curve(y_true, y_scores):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI-generated"], yticklabels=["Human", "AI-generated"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the trained spoof detection model.")
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "best_model.pth"), help="Path to model checkpoint.")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV. If omitted, dataset registry is used.")
    parser.add_argument("--dataset", type=str, default="asvspoof2019", choices=list(DATASET_CONFIG.keys()), help="Dataset to evaluate.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "eval"], help="Dataset split to evaluate.")
    args = parser.parse_args()
    evaluate(args.model, metadata_path=args.metadata, split=args.split, dataset_name=args.dataset)
