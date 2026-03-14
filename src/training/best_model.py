from __future__ import annotations

import json
import sys
from pathlib import Path

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

BEST_PARAMS = {
    "num_layers": 1,
    "start_neurons": 96,
    "dropout": 0.1,
    "lr": 0.0007624212166378842,
    "batch_size": 128,
    "epochs": 25,
}


def train_model(train_tensor: torch.Tensor, params: dict) -> Autoencoder:
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
    for epoch in range(params["epochs"]):
        running_loss = 0.0
        for (batch_data,) in loader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{params['epochs']} - Loss: {avg_loss:.6f}")

    return model


def main() -> None:
    torch.manual_seed(42)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing training tensor and feature artifacts...")
    train_tensor = preprocess_data(DATA_PATH, MODEL_SAVE_DIR).to(DEVICE)

    print("Training best autoencoder configuration...")
    model = train_model(train_tensor, BEST_PARAMS)

    torch.save(model.state_dict(), MODEL_PATH)

    model_config = {
        "input_dim": int(train_tensor.shape[1]),
        "start_neurons": int(BEST_PARAMS["start_neurons"]),
        "dropout": float(BEST_PARAMS["dropout"]),
        "num_layers": int(BEST_PARAMS["num_layers"]),
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    print(f"Saved model weights: {MODEL_PATH}")
    print(f"Saved model config : {MODEL_CONFIG_PATH}")
    print(f"Device used        : {DEVICE}")


if __name__ == "__main__":
    main()
