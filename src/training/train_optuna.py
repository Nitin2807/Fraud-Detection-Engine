from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.processing.preprocessed import preprocess_data
from src.training.model import Autoencoder, DEVICE

DATA_PATH = PROJECT_ROOT / "training_data.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_SAVE_DIR / "best_autoencoder.pth"
MODEL_CONFIG_PATH = MODEL_SAVE_DIR / "model_config.json"

TRAIN_TENSOR: torch.Tensor | None = None


def _train_once(train_tensor: torch.Tensor, params: dict) -> tuple[Autoencoder, float]:
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = Autoencoder(
        input_dim=train_tensor.shape[1],
        output_neurons_layer=params["start_neurons"],
        dropout_rate=params["dropout"],
        num_layers=params["num_layers"],
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for _ in range(params["epochs"]):
        running_loss = 0.0
        for (batch_data,) in loader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        final_loss = running_loss / max(len(loader), 1)

    return model, final_loss


def objective(trial: optuna.Trial) -> float:
    if TRAIN_TENSOR is None:
        raise RuntimeError("TRAIN_TENSOR is not initialized")

    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 4, step=1),
        "start_neurons": trial.suggest_int("start_neurons", 32, 128, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": trial.suggest_int("epochs", 15, 35, step=5),
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        _, final_loss = _train_once(TRAIN_TENSOR, params)
        mlflow.log_metric("final_loss", final_loss)

    return final_loss


def main() -> None:
    global TRAIN_TENSOR
    torch.manual_seed(42)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing training tensor and feature artifacts...")
    TRAIN_TENSOR = preprocess_data(DATA_PATH, MODEL_SAVE_DIR).to(DEVICE)

    mlflow.set_experiment("Fraud_Detection_v2")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_model, best_loss = _train_once(TRAIN_TENSOR, best_params)

    torch.save(best_model.state_dict(), MODEL_PATH)

    model_config = {
        "input_dim": int(TRAIN_TENSOR.shape[1]),
        "start_neurons": int(best_params["start_neurons"]),
        "dropout": float(best_params["dropout"]),
        "num_layers": int(best_params["num_layers"]),
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    print("Best trial parameters:")
    print(best_params)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved model weights: {MODEL_PATH}")
    print(f"Saved model config : {MODEL_CONFIG_PATH}")


if __name__ == "__main__":
    main()
