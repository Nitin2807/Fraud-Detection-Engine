from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.model import Autoencoder
from src.utils.feature_pipeline import transform_with_artifacts

DATA_PATH = PROJECT_ROOT / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "std_scaler.bin"
COLUMNS_PATH = PROJECT_ROOT / "models" / "model_columns.bin"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"

TARGET_RECALL = 0.80
HOLDOUT_SIZE = 0.30
RANDOM_STATE = 42

DEFAULT_MODEL_CONFIG = {
    "start_neurons": 96,
    "dropout": 0.1,
    "num_layers": 1,
}


def _load_model_config() -> dict:
    if not MODEL_CONFIG_PATH.exists():
        return DEFAULT_MODEL_CONFIG.copy()

    payload = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    merged = DEFAULT_MODEL_CONFIG.copy()
    merged.update(payload)
    return merged


def _torch_load_state_dict(path: Path):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def compute_scores(model: Autoencoder, feature_matrix: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor_data = torch.from_numpy(feature_matrix)
        reconstructions = model(tensor_data)
        losses = torch.mean((reconstructions - tensor_data) ** 2, dim=1)
    return losses.numpy()


def load_scaler_with_refresh(path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        scaler = joblib.load(path)

    if any(isinstance(w.message, InconsistentVersionWarning) for w in caught):
        joblib.dump(scaler, path)
        print(f"Refreshed scaler artifact for current sklearn runtime: {path}")

    return scaler


def sweep_thresholds(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> tuple[float, pd.DataFrame]:
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 500)))
    rows = []

    for threshold in candidate_thresholds:
        y_pred = (scores >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    candidates = sweep_df[sweep_df["recall"] >= target_recall]
    if not candidates.empty:
        selected = candidates.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        selected = sweep_df.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0]

    return float(selected["threshold"]), sweep_df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required_paths = [DATA_PATH, MODEL_PATH, SCALER_PATH, COLUMNS_PATH]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_PATH)
    if "is_anomaly" not in df.columns:
        raise ValueError("Dataset must include `is_anomaly` column for evaluation")

    y = df["is_anomaly"].astype(int)
    _, holdout_idx = train_test_split(
        np.arange(len(df)), test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    holdout_df = df.iloc[holdout_idx].copy().reset_index(drop=True)
    y_holdout = holdout_df["is_anomaly"].astype(int).to_numpy()

    scaler = load_scaler_with_refresh(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)

    feature_df = transform_with_artifacts(holdout_df, scaler, feature_columns)
    feature_matrix = feature_df.to_numpy(dtype="float32")

    model_config = _load_model_config()
    model = Autoencoder(
        input_dim=len(feature_columns),
        output_neurons_layer=int(model_config["start_neurons"]),
        dropout_rate=float(model_config["dropout"]),
        num_layers=int(model_config["num_layers"]),
    )
    model.load_state_dict(_torch_load_state_dict(MODEL_PATH))
    model.eval()

    scores = compute_scores(model, feature_matrix)

    selected_threshold, sweep_df = sweep_thresholds(y_holdout, scores, TARGET_RECALL)
    y_pred = (scores >= selected_threshold).astype(int)

    precision = precision_score(y_holdout, y_pred, zero_division=0)
    recall = recall_score(y_holdout, y_pred, zero_division=0)
    f1 = f1_score(y_holdout, y_pred, zero_division=0)
    accuracy = accuracy_score(y_holdout, y_pred)
    pr_auc = average_precision_score(y_holdout, scores)

    try:
        roc_auc = roc_auc_score(y_holdout, scores)
    except ValueError:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred, labels=[0, 1]).ravel()

    threshold_payload = {
        "method": "recall_constrained",
        "target_recall": TARGET_RECALL,
        "selected_threshold": float(selected_threshold),
        "high_risk_multiplier": 1.5,
        "holdout_size": HOLDOUT_SIZE,
        "random_state": RANDOM_STATE,
    }

    metrics_payload = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "pr_auc": float(pr_auc),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "sample_counts": {
            "holdout_rows": int(len(holdout_df)),
            "normal": int((y_holdout == 0).sum()),
            "fraud": int((y_holdout == 1).sum()),
        },
    }

    display_cols = [
        "transactionId",
        "timestamp",
        "type",
        "amount",
        "originCountry",
        "destCountry",
        "is_anomaly",
    ]
    display_cols = [col for col in display_cols if col in holdout_df.columns]
    scored_df = holdout_df[display_cols].copy()
    scored_df["anomaly_score"] = scores
    scored_df["predicted_label"] = y_pred

    (RESULTS_DIR / "threshold.json").write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    sweep_df.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    scored_df.to_csv(RESULTS_DIR / "scored_holdout.csv", index=False)

    summary = (
        "# Evaluation Summary\n\n"
        "## Threshold Selection\n"
        f"- Method: recall-constrained\n"
        f"- Target recall: {TARGET_RECALL:.2f}\n"
        f"- Selected threshold: {selected_threshold:.6f}\n"
        f"- High-risk threshold: {selected_threshold * 1.5:.6f}\n\n"
        "## Metrics (Holdout)\n"
        f"- Precision: {precision:.4f}\n"
        f"- Recall: {recall:.4f}\n"
        f"- F1: {f1:.4f}\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- PR-AUC: {pr_auc:.4f}\n"
        f"- ROC-AUC: {('N/A' if roc_auc is None else f'{roc_auc:.4f}')}\n\n"
        "## Confusion Matrix\n"
        f"- TN: {tn}\n"
        f"- FP: {fp}\n"
        f"- FN: {fn}\n"
        f"- TP: {tp}\n"
    )
    (RESULTS_DIR / "metrics_summary.md").write_text(summary, encoding="utf-8")

    print("Saved:")
    print(f"- {RESULTS_DIR / 'threshold.json'}")
    print(f"- {RESULTS_DIR / 'metrics.json'}")
    print(f"- {RESULTS_DIR / 'threshold_sweep.csv'}")
    print(f"- {RESULTS_DIR / 'scored_holdout.csv'}")
    print(f"- {RESULTS_DIR / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()

