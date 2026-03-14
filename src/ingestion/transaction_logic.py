from __future__ import annotations

import random
import uuid
from datetime import datetime

from faker import Faker

COUNTRIES_TRUSTED = ["India", "USA", "UK", "Germany", "Japan", "Singapore"]
COUNTRIES_HIGH_RISK = ["Panama", "Russia", "Cayman Islands", "North Korea", "Nigeria"]

CURRENCIES_TRUSTED = ["INR", "USD", "GBP", "EUR", "JPY", "SGD"]
CURRENCIES_RISKY = ["RUB", "XMR", "NGN", "BTC"]

BANKS_TRUSTED = [
    "HDFC",
    "SBI",
    "ICICI",
    "Chase",
    "Barclays",
    "Deutsche Bank",
    "HSBC",
    "Mitsubishi UFJ",
]
BANKS_SHADY = ["Cayman Offshore", "Panama Shell", "Silence LLC", "CryptoVault", "DarkWeb Holdings"]

TXN_TYPES = ["CASH_IN", "CASH_OUT"]
ANOMALY_FACTORS = ["huge_amount", "shady_bank", "high_risk_country", "risky_currency", "math_error"]

CURRENCY_BY_COUNTRY = {
    "India": "INR",
    "USA": "USD",
    "UK": "GBP",
    "Germany": "EUR",
    "Japan": "JPY",
    "Singapore": "SGD",
}


def _base_currency_for_country(country: str) -> str:
    return CURRENCY_BY_COUNTRY.get(country, "USD")


def _build_balances(txn_type: str, amount: float) -> tuple[float, float, float, float]:
    if txn_type == "CASH_OUT":
        old_balance_org = round(random.uniform(amount * 1.1, 1_000_000.0), 2)
        new_balance_org = round(old_balance_org - amount, 2)
        old_balance_dest = round(random.uniform(0.0, 500_000.0), 2)
        new_balance_dest = round(old_balance_dest + amount, 2)
    else:
        old_balance_org = round(random.uniform(0.0, 500_000.0), 2)
        new_balance_org = round(old_balance_org + amount, 2)
        old_balance_dest = round(random.uniform(amount * 1.1, 1_000_000.0), 2)
        new_balance_dest = round(old_balance_dest - amount, 2)

    return old_balance_org, new_balance_org, old_balance_dest, new_balance_dest


def generate_transaction(fake: Faker, anomaly_rate: float = 0.05, include_label: bool = False) -> dict:
    is_anomaly = random.random() < anomaly_rate
    txn_type = random.choice(TXN_TYPES)

    src_country = random.choice(COUNTRIES_TRUSTED)
    if random.random() < 0.8:
        dest_country = src_country
    else:
        dest_country = random.choice(COUNTRIES_TRUSTED)

    src_bank = random.choice(BANKS_TRUSTED)
    dest_bank = random.choice(BANKS_TRUSTED)
    currency = _base_currency_for_country(src_country)

    amount = round(random.uniform(50.0, 50_000.0), 2)
    math_error = False

    if is_anomaly:
        risk_factor = random.choice(ANOMALY_FACTORS)

        if risk_factor == "huge_amount":
            amount = round(random.uniform(200_000.0, 5_000_000.0), 2)
        elif risk_factor == "shady_bank":
            dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == "high_risk_country":
            dest_country = random.choice(COUNTRIES_HIGH_RISK)
            if random.random() < 0.5:
                dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == "risky_currency":
            currency = random.choice(CURRENCIES_RISKY)
        elif risk_factor == "math_error":
            math_error = True

    old_balance_org, new_balance_org, old_balance_dest, new_balance_dest = _build_balances(txn_type, amount)

    if math_error:
        if txn_type == "CASH_OUT":
            new_balance_org = old_balance_org
        else:
            new_balance_dest = old_balance_dest

    payload = {
        "transactionId": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "type": txn_type,
        "amount": amount,
        "currency": currency,
        "originBank": src_bank,
        "originLocation": f"{fake.city()}, {src_country}",
        "originCountry": src_country,
        "oldBalanceOrg": old_balance_org,
        "newBalanceOrg": new_balance_org,
        "destBank": dest_bank,
        "destLocation": f"{fake.city()}, {dest_country}",
        "destCountry": dest_country,
        "oldBalanceDest": old_balance_dest,
        "newBalanceDest": new_balance_dest,
    }

    if include_label:
        payload["is_anomaly"] = int(is_anomaly)

    return payload


def _parse_amount(value) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def is_transaction_suspicious(transaction: dict) -> bool:
    amount = _parse_amount(transaction.get("amount", 0.0))
    txn_type = str(transaction.get("type", "")).upper()

    old_org = _parse_amount(transaction.get("oldBalanceOrg", 0.0))
    new_org = _parse_amount(transaction.get("newBalanceOrg", 0.0))
    old_dest = _parse_amount(transaction.get("oldBalanceDest", 0.0))
    new_dest = _parse_amount(transaction.get("newBalanceDest", 0.0))

    if txn_type == "CASH_OUT":
        balance_mismatch = abs((old_org - amount) - new_org) > 1.0 or abs((old_dest + amount) - new_dest) > 1.0
    elif txn_type == "CASH_IN":
        balance_mismatch = abs((old_org + amount) - new_org) > 1.0 or abs((old_dest - amount) - new_dest) > 1.0
    else:
        balance_mismatch = False

    return any(
        [
            transaction.get("destBank") in BANKS_SHADY,
            transaction.get("destCountry") in COUNTRIES_HIGH_RISK,
            transaction.get("currency") in CURRENCIES_RISKY,
            amount > 100_000.0,
            balance_mismatch,
        ]
    )
