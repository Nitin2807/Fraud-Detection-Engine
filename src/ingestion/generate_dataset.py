from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ingestion.transaction_logic import generate_transaction


NUM_RECORDS = 100_000
OUTPUT_FILE = PROJECT_ROOT / "training_data.csv"
fake = Faker()


def inject_nulls(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, "destLocation"] = np.nan
    df.loc[mask, "originCountry"] = np.nan
    df.loc[mask, "timestamp"] = np.nan
    return df


def inject_casing_errors(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.random.rand(len(df)) < 0.10
    df.loc[mask, "originBank"] = df.loc[mask, "originBank"].str.lower()
    return df


def inject_type_errors(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"] = df["amount"].astype(object)
    mask = np.random.rand(len(df)) < 0.15
    df.loc[mask, "amount"] = df.loc[mask, "amount"].apply(lambda x: f"{x:,.2f}")
    return df


def inject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df.sample(frac=0.08, random_state=42)
    return pd.concat([df, duplicates], ignore_index=True)


def main() -> None:
    print(f"Step 1: Generating {NUM_RECORDS} clean records with type-aware balance math...")
    data = [generate_transaction(fake=fake, include_label=True) for _ in range(NUM_RECORDS)]
    df = pd.DataFrame(data)

    print("Step 2: Injecting data quality noise...")
    df = inject_nulls(df)
    df = inject_casing_errors(df)
    df = inject_type_errors(df)
    df = inject_duplicates(df)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved dataset: {OUTPUT_FILE}")

    type_distribution = df["type"].value_counts(normalize=True).mul(100).round(2)
    print("Type distribution (%):")
    for txn_type, pct in type_distribution.items():
        print(f"- {txn_type}: {pct}%")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
