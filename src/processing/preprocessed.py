from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.utils.feature_pipeline import fit_feature_artifacts, save_feature_artifacts


def preprocess_data(data_path: str | Path, model_save_dir: str | Path) -> torch.Tensor:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    feature_df, labels, scaler, feature_columns = fit_feature_artifacts(df)

    if labels is None:
        raise ValueError("Dataset must include `is_anomaly` for autoencoder training")

    save_feature_artifacts(scaler, feature_columns, model_save_dir)

    normal_df = feature_df[labels == 0]
    return torch.from_numpy(normal_df.to_numpy(dtype="float32"))
