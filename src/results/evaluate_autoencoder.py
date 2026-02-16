import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
DATA_PATH = PROJECT_ROOT / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "std_scaler.bin"
COLUMNS_PATH = PROJECT_ROOT / "models" / "model_columns.bin"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"

TARGET_RECALL = 0.90
HOLDOUT_SIZE = 0.30
RANDOM_STATE = 42


class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_neurons_layer, dropout_rate, num_layers):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        current_dim = input_dim
        next_dim = output_neurons_layer
        self.layer_sizes = []
        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.BatchNorm1d(next_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            self.layer_sizes.append(next_dim)
            current_dim = next_dim
            next_dim = max(5, next_dim // 2)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reverse_sizes = self.layer_sizes[::-1]
        current_dim = self.layer_sizes[-1]
        target_sizes = reverse_sizes[1:] + [input_dim]
        for target_dim in target_sizes:
            decoder_layers.append(nn.Linear(current_dim, target_dim))
            if target_dim != num_layers - 1:
                decoder_layers.append(nn.BatchNorm1d(target_dim))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = target_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_feature_matrix(df: pd.DataFrame, scaler, feature_columns: pd.Index) -> np.ndarray:
    num_cols = ["amount", "oldBalanceOrg", "newBalanceOrg", "oldBalanceDest", "newBalanceDest"]
    cat_cols = ["type", "currency", "originBank", "originCountry", "destBank", "destCountry"]

    work_df = df.copy()
    work_df["amount"] = work_df["amount"].astype(str).str.replace(",", "", regex=False).astype(float)
    work_df["originBank"] = work_df["originBank"].astype(str).str.upper()
    work_df.fillna("UNKNOWN", inplace=True)

    encoded = pd.get_dummies(work_df[cat_cols], drop_first=True)
    feature_df = encoded.copy()
    for col_name in num_cols:
        feature_df[col_name] = work_df[col_name].astype(float)

    feature_df = feature_df.reindex(columns=feature_columns, fill_value=0.0)
    feature_df[num_cols] = scaler.transform(feature_df[num_cols])
    return feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")


def compute_scores(model: nn.Module, feature_matrix: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor_data = torch.from_numpy(feature_matrix)
        reconstructions = model(tensor_data)
        losses = torch.mean((reconstructions - tensor_data) ** 2, dim=1)
    return losses.numpy()


def load_scaler_with_refresh(path: Path):
    # If sklearn version changed, load once and re-save to current runtime version.
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "is_anomaly" not in df.columns:
        raise ValueError("Dataset must include `is_anomaly` column for evaluation")

    eval_df = df.copy()
    y = eval_df["is_anomaly"].astype(int)

    # This creates a holdout split for evaluation artifacts.
    _, holdout_idx = train_test_split(
        np.arange(len(eval_df)), test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    holdout_df = eval_df.iloc[holdout_idx].copy().reset_index(drop=True)
    y_holdout = holdout_df["is_anomaly"].astype(int).to_numpy()

    scaler = load_scaler_with_refresh(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)

    feature_matrix = build_feature_matrix(holdout_df, scaler, feature_columns)

    model = Autoencoder(
        input_dim=len(feature_columns), output_neurons_layer=96, dropout_rate=0.1, num_layers=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
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

    scored_df = holdout_df[["transactionId", "timestamp", "type", "amount", "originCountry", "destCountry", "is_anomaly"]].copy()
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
        f"- Selected threshold: {selected_threshold:.6f}\n\n"
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
