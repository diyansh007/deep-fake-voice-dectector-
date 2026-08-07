import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def label_to_int(label: str, mapping: dict):
    normalized = str(label).strip().lower()
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unknown label '{label}' for mapping.")


def int_to_label(index: int):
    return "Human" if index == 0 else "AI-generated"


def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_precision():
        if fp + tp == 0:
            return None
        return float(precision_score(y_true, y_pred, zero_division=0))

    def safe_recall():
        if fn + tp == 0:
            return None
        return float(recall_score(y_true, y_pred, zero_division=0))

    precision = safe_precision()
    recall = safe_recall()
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
        "eer": None,
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
        if metrics["roc_auc"] is not None:
            try:
                metrics["eer"] = compute_eer(y_true, y_prob[:, 1])
            except Exception:
                metrics["eer"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def risk_from_probability(probability: float):
    if probability >= 0.9:
        return "High"
    if probability >= 0.7:
        return "Medium"
    return "Low"
