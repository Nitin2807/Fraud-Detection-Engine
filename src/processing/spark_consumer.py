from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import os
import re
import subprocess
import sys
from typing import Iterator

# --- CONFIGURATION ---
OUTPUT_MODE = "console" 
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "financial_transactions"
MONGO_URI = "mongodb+srv://<YOUR_USER>:<YOUR_PASSWORD>@<YOUR_CLUSTER>.mongodb.net/?retryWrites=true&w=majority"
MONGO_DB = "fraud_detection"
MONGO_COLLECTION = "transactions"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_autoencoder.pth"
SCALER_PATH = PROJECT_ROOT / "models" / "std_scaler.bin"
COLUMNS_PATH = PROJECT_ROOT / "models" / "model_columns.bin"

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


def _prepare_runtime():
    # Some Windows setups accidentally store HADOOP_HOME as "%HADOOP_HOME%\bin";
    # remove that invalid value so Spark doesn't try to use it.
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if "%" in hadoop_home:
        os.environ.pop("HADOOP_HOME", None)

    # Spark 3.5.x is not compatible with very new JDKs (e.g., 21+ / 25).
    java_major = _get_java_major()
    if java_major == -1:
        raise RuntimeError("Java was not found. Install Java 17 and set JAVA_HOME accordingly.")
    if java_major > 17:
        raise RuntimeError(
            f"Detected Java {java_major}. Please use Java 17 for Spark 3.5. "
            "Set JAVA_HOME to a Java 17 installation and restart the terminal."
        )


def _build_spark_session() -> SparkSession:
    packages = ["org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"]
    if OUTPUT_MODE == "mongodb":
        packages.append("org.mongodb.spark:mongo-spark-connector_2.12:10.2.1")

    return SparkSession.builder \
        .appName("FraudDetectionEngine") \
        .config("spark.jars.packages", ",".join(packages)) \
        .getOrCreate()


_prepare_runtime()
spark = _build_spark_session()

spark.sparkContext.setLogLevel("ERROR")

# --- MODEL CLASS ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_neurons_layer, dropout_rate, num_layers):
        super(Autoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        next_dim = output_neurons_layer
        self.layer_sizes = [] 
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.BatchNorm1d(next_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            self.layer_sizes.append(next_dim)
            current_dim = next_dim
            next_dim = max(5, next_dim // 2) 
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        reverse_sizes = self.layer_sizes[::-1] 
        current_dim = self.layer_sizes[-1] 
        target_sizes = reverse_sizes[1:] + [input_dim]
        for target_dim in target_sizes:
            decoder_layers.append(nn.Linear(current_dim, target_dim))
            if target_dim != input_dim:
                decoder_layers.append(nn.BatchNorm1d(target_dim))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = target_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

_model = None
_scaler = None
_feature_columns = None


def _load_inference_artifacts():
    global _model, _scaler, _feature_columns
    if _model is not None:
        return

    device = torch.device("cpu")
    _scaler = joblib.load(SCALER_PATH)
    _feature_columns = joblib.load(COLUMNS_PATH)
    input_dim = len(_feature_columns)

    _model = Autoencoder(input_dim=input_dim, output_neurons_layer=96, dropout_rate=0.1, num_layers=1)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _model.eval()


def score_batches(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    _load_inference_artifacts()
    num_cols = ['amount', 'oldBalanceOrg', 'newBalanceOrg', 'oldBalanceDest', 'newBalanceDest']
    cat_cols = ['type', 'currency', 'originBank', 'originCountry', 'destBank', 'destCountry']

    for batch_df in iterator:
        if batch_df.empty:
            batch_df["anomaly_score"] = []
            yield batch_df
            continue

        # Data cleaning to match training pipeline.
        batch_df = batch_df.copy()
        batch_df['amount'] = batch_df['amount'].astype(str).str.replace(',', '', regex=False).astype(float)
        batch_df['originBank'] = batch_df['originBank'].astype(str).str.upper()
        batch_df.fillna("UNKNOWN", inplace=True)

        # Keep only model features and align strictly to training columns.
        current_dummies = pd.get_dummies(batch_df[cat_cols], drop_first=True)
        feature_df = current_dummies.copy()
        for col_name in num_cols:
            feature_df[col_name] = batch_df[col_name].astype(float)
        feature_df = feature_df.reindex(columns=_feature_columns, fill_value=0.0)
        feature_df[num_cols] = _scaler.transform(feature_df[num_cols])

        with torch.no_grad():
            tensor_data = torch.FloatTensor(feature_df.values)
            reconstructions = _model(tensor_data)
            loss = torch.mean((reconstructions - tensor_data) ** 2, dim=1).numpy()

        batch_df["anomaly_score"] = loss
        yield batch_df

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    # 1. READ KAFKA
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    # 2. PARSE JSON
    schema = StructType([
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
        StructField("timestamp", StringType())
    ])
    
    df_parsed = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    # 3. APPLY AI MODEL
    output_schema = StructType(schema.fields + [StructField("anomaly_score", DoubleType())])
    df_scored = df_parsed.mapInPandas(score_batches, schema=output_schema)

    # 4. WRITE STREAM
    if OUTPUT_MODE == "console":
        print("🚀 Writing to Console...")
        query = df_scored.writeStream \
            .outputMode("append") \
            .format("console") \
            .start()
            
    elif OUTPUT_MODE == "mongodb":
        print(f"🚀 Writing to MongoDB Atlas: {MONGO_DB}.{MONGO_COLLECTION}")
        query = df_scored.writeStream \
            .format("mongodb") \
            .option("checkpointLocation", "/tmp/pyspark_checkpoint") \
            .option("spark.mongodb.connection.uri", MONGO_URI) \
            .option("spark.mongodb.database", MONGO_DB) \
            .option("spark.mongodb.collection", MONGO_COLLECTION) \
            .outputMode("append") \
            .start()

    query.awaitTermination()
