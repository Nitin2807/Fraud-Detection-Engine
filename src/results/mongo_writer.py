import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# 1. SETUP PATHS (So we can import from 'src')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.processing.spark_consumer import get_scored_stream

# --- CONFIG ---
load_dotenv(PROJECT_ROOT / ".env")
MONGO_URI = os.getenv("uri")
if not MONGO_URI:
    raise RuntimeError("Missing `uri` in .env. Add `uri=<mongodb_connection_string>` to project .env")

DB_NAME = "fraud_detection_engine"
COLLECTION_NAME = "alerts"
CHECKPOINT_DIR = f"file:///{(PROJECT_ROOT / 'spark_checkpoints').as_posix()}"


def validate_mongo_connection(uri: str) -> None:
    client = MongoClient(uri, serverSelectionTimeoutMS=7000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise RuntimeError(
            "MongoDB connection/auth failed. Check .env `uri` credentials and Atlas user permissions. "
            f"Details: {exc}"
        )
    finally:
        client.close()


def start_streaming():
    print("Initializing Fraud Detection Stream...")
    validate_mongo_connection(MONGO_URI)
    print("MongoDB ping successful.")

    # 1. Get the DataFrame from our processing module
    df = get_scored_stream()

    print(f"Connecting to MongoDB Atlas: {DB_NAME}.{COLLECTION_NAME}")

    # 2. Write to MongoDB
    query = (
        df.writeStream.format("mongodb")
        .option("checkpointLocation", CHECKPOINT_DIR)
        .option("spark.mongodb.connection.uri", MONGO_URI)
        .option("spark.mongodb.database", DB_NAME)
        .option("spark.mongodb.collection", COLLECTION_NAME)
        .outputMode("append")
        .start()
    )

    print("Stream is running! Listening for Kafka transactions...")
    query.awaitTermination()


if __name__ == "__main__":
    # Ensure checkpoint dir exists or Spark complains on Windows
    os.makedirs(PROJECT_ROOT / "spark_checkpoints", exist_ok=True)
    start_streaming()
