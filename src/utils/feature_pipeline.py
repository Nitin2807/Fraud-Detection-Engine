from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

DROP_COLUMNS = ["transactionId", "timestamp", "originLocation", "destLocation"]
LABEL_COLUMN = "is_anomaly"
CATEGORICAL_COLS = ["type", "currency", "originBank", "originCountry", "destBank", "destCountry"]
NUMERIC_COLS = ["amount", "oldBalanceOrg", "newBalanceOrg", "oldBalanceDest", "newBalanceDest"]


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    if "amount" in work_df.columns:
        work_df["amount"] = pd.to_numeric(
            work_df["amount"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )

    if "originBank" in work_df.columns:
        work_df["originBank"] = work_df["originBank"].astype(str).str.upper()

    for col in CATEGORICAL_COLS:
        if col not in work_df.columns:
            work_df[col] = "UNKNOWN"
        work_df[col] = work_df[col].fillna("UNKNOWN").astype(str)

    for col in NUMERIC_COLS:
        if col not in work_df.columns:
            work_df[col] = 0.0
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce").fillna(0.0).astype(float)

    return work_df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work_df = clean_transactions(df)
    feature_source = work_df.drop(columns=[*DROP_COLUMNS, LABEL_COLUMN], errors="ignore")

    encoded = pd.get_dummies(feature_source[CATEGORICAL_COLS], drop_first=False, dtype=float)
    feature_df = encoded.copy()

    for col in NUMERIC_COLS:
        feature_df[col] = feature_source[col].astype(float)

    return feature_df.astype(float)


def fit_feature_artifacts(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, StandardScaler, list[str]]:
    work_df = clean_transactions(df)

    labels: pd.Series | None = None
    if LABEL_COLUMN in work_df.columns:
        labels = pd.to_numeric(work_df[LABEL_COLUMN], errors="coerce").fillna(0).astype(int)

    feature_df = build_feature_frame(work_df)

    scaler = StandardScaler()
    feature_df.loc[:, NUMERIC_COLS] = scaler.fit_transform(feature_df[NUMERIC_COLS])
    feature_columns = feature_df.columns.tolist()

    return feature_df, labels, scaler, feature_columns


def transform_with_artifacts(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    feature_df = build_feature_frame(df)
    feature_df = feature_df.reindex(columns=list(feature_columns), fill_value=0.0)

    feature_df.loc[:, NUMERIC_COLS] = scaler.transform(feature_df[NUMERIC_COLS])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return feature_df.astype("float32")


def save_feature_artifacts(scaler: StandardScaler, feature_columns: Iterable[str], model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, model_dir / "std_scaler.bin")
    joblib.dump(list(feature_columns), model_dir / "model_columns.bin")
