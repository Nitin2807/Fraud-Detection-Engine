from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterator

import joblib
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, when
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from sklearn.exceptions import InconsistentVersionWarning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.model import Autoencoder
from src.utils.feature_pipeline import transform_with_artifacts


KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "financial_transactions"

MODEL_PATH = PROJECT_ROOT / "models" / "best_autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "std_scaler.bin"
COLUMNS_PATH = PROJECT_ROOT / "models" / "model_columns.bin"
MODEL_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
THRESHOLD_PATH = PROJECT_ROOT / "src" / "results" / "threshold.json"

DEFAULT_ANOMALY_THRESHOLD = 0.10
DEFAULT_MODEL_CONFIG = {
    "start_neurons": 96,
    "dropout": 0.1,
    "num_layers": 1,
}


def _load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        return DEFAULT_ANOMALY_THRESHOLD

    try:
        payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        return float(payload.get("selected_threshold", DEFAULT_ANOMALY_THRESHOLD))
    except Exception:
        return DEFAULT_ANOMALY_THRESHOLD


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


def _get_java_major() -> int:
    java_home = os.environ.get("JAVA_HOME")
    java_cmd = os.path.join(java_home, "bin", "java.exe") if java_home else "java"
    try:
        proc = subprocess.run([java_cmd, "-version"], capture_output=True, text=True, check=False)
    except OSError:
        return -1

    output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r'"(\d+)(?:\.\d+)*"', output)
    return int(match.group(1)) if match else -1


def _prepare_runtime() -> None:
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if "%" in hadoop_home:
        os.environ.pop("HADOOP_HOME", None)

    java_major = _get_java_major()
    if java_major == -1:
        raise RuntimeError("Java was not found. Install Java 17 and set JAVA_HOME accordingly.")
    if java_major > 17:
        raise RuntimeError(
            f"Detected Java {java_major}. Please use Java 17 for Spark 3.5. "
            "Set JAVA_HOME to a Java 17 installation and restart the terminal."
        )


def _build_spark_session() -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            project_root_str if not current_pythonpath else project_root_str + os.pathsep + current_pythonpath
        )

    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    packages = [
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
    ]

    return (
        SparkSession.builder.appName("FraudDetectionEngine")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.ip", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .config("spark.jars.packages", ",".join(packages))
        .getOrCreate()
    )


spark = None


def _get_spark() -> SparkSession:
    global spark
    if spark is None:
        _prepare_runtime()
        spark = _build_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
    return spark


_model = None
_scaler = None
_feature_columns = None


def _load_scaler_with_refresh(path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        scaler = joblib.load(path)
    if any(isinstance(w.message, InconsistentVersionWarning) for w in caught):
        joblib.dump(scaler, path)
    return scaler


def _load_inference_artifacts() -> None:
    global _model, _scaler, _feature_columns

    if _model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file missing: {SCALER_PATH}")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature column file missing: {COLUMNS_PATH}")

    model_config = _load_model_config()

    _scaler = _load_scaler_with_refresh(SCALER_PATH)
    _feature_columns = joblib.load(COLUMNS_PATH)

    _model = Autoencoder(
        input_dim=len(_feature_columns),
        output_neurons_layer=int(model_config["start_neurons"]),
        dropout_rate=float(model_config["dropout"]),
        num_layers=int(model_config["num_layers"]),
    )
    _model.load_state_dict(_torch_load_state_dict(MODEL_PATH))
    _model.eval()


def score_batches(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    _load_inference_artifacts()

    for batch_df in iterator:
        if batch_df.empty:
            empty_df = batch_df.copy()
            empty_df["anomaly_score"] = pd.Series(dtype="float64")
            yield empty_df
            continue

        output_df = batch_df.copy()
        feature_df = transform_with_artifacts(batch_df, _scaler, _feature_columns)
        feature_matrix = feature_df.to_numpy(dtype="float32")

        with torch.no_grad():
            tensor_data = torch.from_numpy(feature_matrix)
            reconstructions = _model(tensor_data)
            loss = torch.mean((reconstructions - tensor_data) ** 2, dim=1).numpy()

        output_df["anomaly_score"] = loss
        yield output_df


def get_scored_stream():
    spark_session = _get_spark()

    df_raw = (
        spark_session.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    schema = StructType(
        [
            StructField("transactionId", StringType()),
            StructField("amount", StringType()),
            StructField("type", StringType()),
            StructField("currency", StringType()),
            StructField("oldBalanceOrg", DoubleType()),
            StructField("newBalanceOrg", DoubleType()),
            StructField("oldBalanceDest", DoubleType()),
            StructField("newBalanceDest", DoubleType()),
            StructField("originBank", StringType()),
            StructField("destBank", StringType()),
            StructField("originCountry", StringType()),
            StructField("destCountry", StringType()),
            StructField("timestamp", StringType()),
        ]
    )

    df_parsed = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    output_schema = StructType(schema.fields + [StructField("anomaly_score", DoubleType())])
    df_scored = df_parsed.mapInPandas(score_batches, schema=output_schema)

    threshold = _load_threshold()
    high_threshold = threshold * 1.5

    df_scored = (
        df_scored.withColumn("is_fraud_transaction", (col("anomaly_score") >= lit(high_threshold)).cast("int"))
        .withColumn("is_suspected_fraud", (col("anomaly_score") >= lit(high_threshold)).cast("int"))
        .withColumn(
            "risk_band",
            when(col("anomaly_score") >= lit(high_threshold), lit("high"))
            .when(col("anomaly_score") >= lit(threshold), lit("medium"))
            .otherwise(lit("low")),
        )
    )

    return df_scored
