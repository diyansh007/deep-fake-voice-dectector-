import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import TRAINING_CONFIG, MODEL_DIR, PLOTS_DIR, RANDOM_SEED, FEATURE_TYPE, DATA_LOADER_WORKERS, DATASET_CONFIG
from utils import set_seed, setup_logger, compute_metrics
from datasets.metadata_builder import get_or_build_metadata
from datasets.dataset import SpoofAudioDataset
from preprocessing.augmentation import augment_audio
from models.optimized_cnn import SpectrogramResNet
import matplotlib.pyplot as plt
import seaborn as sns


def build_data_loaders(metadata: pd.DataFrame, batch_size: int, feature_type: str = FEATURE_TYPE, num_workers: int = DATA_LOADER_WORKERS):
    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta = metadata[metadata["split"] == "dev"].reset_index(drop=True)
    if val_meta.empty:
        train_meta, val_meta = train_test_split(
            train_meta,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=train_meta["label"],
        )
        val_split = None
    else:
        val_split = "dev"
    train_dataset = SpoofAudioDataset(train_meta, split="train", feature_type=feature_type, augment_fn=augment_audio)
    val_dataset = SpoofAudioDataset(val_meta, split=val_split, feature_type=feature_type, augment_fn=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def train(epochs: int, batch_size: int, dataset_name: str, force_rebuild: bool):
    set_seed(RANDOM_SEED)
    logger = setup_logger("train", os.path.join(MODEL_DIR, "train.log"))
    metadata = get_or_build_metadata(dataset_name, force_rebuild=force_rebuild)
    logger.info(f"Metadata loaded with {len(metadata)} rows for dataset {dataset_name}.")
    class_counts = metadata[metadata["split"] == "train"]["label"].value_counts().to_dict()
    logger.info(f"Train split class distribution: {class_counts}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = SpectrogramResNet(num_classes=2, in_channels=3 if FEATURE_TYPE == "log_mel_delta" else 1).to(device)

    # Select checkpoint based on dataset
    if dataset_name == "fake_or_real":
        checkpoint_path = os.path.join(MODEL_DIR, "best_model_for.pth")

        # If this is the first Fake-or-Real run, fall back to the ASVspoof model
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")
    else:
        checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No pretrained checkpoint found. Training from scratch.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=TRAINING_CONFIG["scheduler_factor"],
        patience=TRAINING_CONFIG["scheduler_patience"],
        min_lr=TRAINING_CONFIG["min_lr"],
    )
    train_labels = metadata[metadata["split"] == "train"]["label"].values
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        [len(train_labels) / (2.0 * c) if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(MODEL_DIR, "tensorboard"))
    best_val_score = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "lr": [],
    }

    train_loader, val_loader = build_data_loaders(metadata, batch_size, feature_type=FEATURE_TYPE)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda" or device.type == "mps")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_val"])
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        val_losses = []
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                probabilities = torch.softmax(outputs, dim=1)
                preds = probabilities.argmax(dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
                y_prob.extend(probabilities.cpu().numpy().tolist())

        y_prob_tensor = torch.tensor(y_prob)
        metrics = compute_metrics(y_true, y_pred, y_prob=y_prob_tensor)
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(metrics["accuracy"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])
        history["roc_auc"].append(metrics.get("roc_auc", 0.0))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", metrics["accuracy"], epoch)
        writer.add_scalar("Precision/val", metrics["precision"], epoch)
        writer.add_scalar("Recall/val", metrics["recall"], epoch)
        writer.add_scalar("F1/val", metrics["f1"], epoch)
        writer.add_scalar("ROC_AUC/val", metrics.get("roc_auc", 0.0), epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        scheduler.step(metrics.get("roc_auc", metrics["f1"]))
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics.get('roc_auc', 0.0):.4f}, fp={metrics['fp']}, fn={metrics['fn']}"
        )

        current_score = metrics.get(TRAINING_CONFIG["early_stop_monitor"], metrics["f1"])
        if current_score > best_val_score:
            best_val_score = current_score
            patience_counter = 0
            if dataset_name == "fake_or_real":
                best_model_path = os.path.join(MODEL_DIR, "best_model_for.pth")
            else:
                best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model at epoch {epoch} with best_{TRAINING_CONFIG['early_stop_monitor']}={best_val_score:.4f} to {best_model_path}.")
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG["patience"]:
                logger.info("Early stopping triggered.")
                break

    if dataset_name == "fake_or_real":
        last_model_path = os.path.join(MODEL_DIR, "last_model_for.pth")
    else:
        last_model_path = os.path.join(MODEL_DIR, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    writer.close()
    logger.info(f"Saved last model and history to {MODEL_DIR}.")
    plot_history(history)


def plot_history(history: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history["accuracy"], label="accuracy")
    plt.plot(history["precision"], label="precision")
    plt.plot(history["recall"], label="recall")
    plt.plot(history["f1"], label="f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Metrics")
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_curve.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the spoof detection model.")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"], help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"], help="Training batch size.")
    parser.add_argument("--dataset", type=str, default="asvspoof2019", choices=list(DATASET_CONFIG.keys()), help="Dataset to build metadata and train on.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild metadata even if cached metadata exists.")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.dataset, args.force_rebuild)
