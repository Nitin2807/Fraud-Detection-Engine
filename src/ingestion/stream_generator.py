from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from faker import Faker
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ingestion.transaction_logic import generate_transaction, is_transaction_suspicious


KAFKA_TOPIC = "financial_transactions"
BROKER = "localhost:9092"

fake = Faker()


def inject_data_dirt(data: dict) -> dict:
    dirt_roll = random.random()

    if dirt_roll < 0.10:
        data["originBank"] = str(data["originBank"]).lower()
    elif dirt_roll < 0.20:
        data["amount"] = f"{float(data['amount']):,.2f}"
    elif dirt_roll < 0.25:
        data["destLocation"] = None
    elif dirt_roll < 0.30:
        data["extra_garbage_field"] = "IGNORE_ME"

    return data


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
    )


def main() -> None:
    try:
        producer = build_producer()
        print(f"Connected to Kafka at {BROKER}")
    except Exception as exc:
        print(f"Failed to connect to Kafka: {exc}")
        return

    print(f"Streaming transactions to topic: {KAFKA_TOPIC}")

    try:
        while True:
            txn = generate_transaction(fake=fake, include_label=False)
            suspicious = is_transaction_suspicious(txn)

            dirty_txn = inject_data_dirt(dict(txn))
            producer.send(KAFKA_TOPIC, value=dirty_txn)

            amount_text = str(dirty_txn.get("amount", ""))
            route_text = f"{dirty_txn.get('originCountry')} -> {dirty_txn.get('destCountry')}"
            if suspicious:
                print(f"[ALERT] {dirty_txn.get('currency')} {amount_text.rjust(12)} | {route_text} | {dirty_txn.get('type')}")
            else:
                print(f"[OK]    {dirty_txn.get('currency')} {amount_text.rjust(12)} | {route_text} | {dirty_txn.get('type')}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stream stopped.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
