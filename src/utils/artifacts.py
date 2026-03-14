from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.feature_pipeline import fit_feature_artifacts, save_feature_artifacts

DATA_PATH = PROJECT_ROOT / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    feature_df, labels, scaler, feature_columns = fit_feature_artifacts(df)
    save_feature_artifacts(scaler, feature_columns, MODEL_DIR)

    normal_rows = int((labels == 0).sum()) if labels is not None else len(df)

    print(f"Saved scaler to: {MODEL_DIR / 'std_scaler.bin'}")
    print(f"Saved columns to: {MODEL_DIR / 'model_columns.bin'}")
    print(f"Rows processed : {len(df)}")
    print(f"Normal rows    : {normal_rows}")
    print(f"Feature count  : {feature_df.shape[1]}")


if __name__ == "__main__":
    main()
