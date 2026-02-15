import time
import json
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

# Setup
KAFKA_TOPIC = "financial_transactions"
BROKER = 'localhost:9092'

fake = Faker()

try:
    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print(f"✅ Connected to Kafka at {BROKER}")
except Exception as e:
    print(f"❌ Failed to connect to Kafka: {e}")
    exit()

# 1. Geographic Profiles
COUNTRIES_TRUSTED = ['India', 'USA', 'UK', 'Germany', 'Japan', 'Singapore']
COUNTRIES_HIGH_RISK = ['Panama', 'Russia', 'Cayman Islands', 'North Korea', 'Nigeria']

# 2. Currency Profiles
CURRENCIES_TRUSTED = ['INR', 'USD', 'GBP', 'EUR', 'JPY', 'SGD']
CURRENCIES_RISKY = ['RUB', 'XMR', 'NGN', 'BTC'] 

# 3. Bank Profiles
BANKS_TRUSTED = ['HDFC', 'SBI', 'ICICI', 'Chase', 'Barclays', 'Deutsche Bank', 'HSBC', 'Mitsubishi UFJ']
BANKS_SHADY = ['Cayman Offshore', 'Panama Shell', 'Silence LLC', 'CryptoVault', 'DarkWeb Holdings']

TXN_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT']

def generate_transaction():
    txn_id = str(uuid.uuid4())
    is_anomaly = random.random() < 0.05 # 5% Anomaly Rate
    
    # --- DEFAULT: Normal Behavior ---
    src_country = random.choice(COUNTRIES_TRUSTED)
    
    # 80% chance Domestic, 20% International
    if random.random() < 0.8:
        dest_country = src_country
    else:
        dest_country = random.choice(COUNTRIES_TRUSTED)
        
    src_bank = random.choice(BANKS_TRUSTED)
    dest_bank = random.choice(BANKS_TRUSTED)
    
    # Simple Currency Logic
    currency = "USD" if src_country == "USA" else "INR" if src_country == "India" else "EUR"
    amount = round(random.uniform(50.0, 50000.0), 2)
    math_error = False

    # --- ANOMALY INJECTION ---
    if is_anomaly:
        risk_factor = random.choice(['huge_amount', 'shady_bank', 'high_risk_country', 'risky_currency', 'math_error'])
        
        if risk_factor == 'huge_amount':
            amount = round(random.uniform(200000.0, 5000000.0), 2)
        elif risk_factor == 'shady_bank':
            dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == 'high_risk_country':
            dest_country = random.choice(COUNTRIES_HIGH_RISK)
            if random.random() < 0.5:
                dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == 'risky_currency':
            currency = random.choice(CURRENCIES_RISKY)
        elif risk_factor == 'math_error':
            math_error = True

    # --- Math Logic ---
    old_balance_org = round(random.uniform(amount * 1.1, 1000000.0), 2)
    new_balance_org = round(old_balance_org - amount, 2)
    
    old_balance_dest = round(random.uniform(0, 500000.0), 2)
    new_balance_dest = round(old_balance_dest + amount, 2)

    if math_error:
        new_balance_org = old_balance_org 

    # Construct Payload
    data = {
        "transactionId": txn_id,
        "timestamp": datetime.now().isoformat(),
        "type": random.choice(TXN_TYPES),
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
        "newBalanceDest": new_balance_dest
    }

    return data

def inject_data_dirt(data):
    """
    The 'Spark Job Security' Function.
    Injects formatting errors that Spark MUST fix.
    """
    dirt_roll = random.random()

    # 1. Casing Issues (Spark needs to .upper() this)
    if dirt_roll < 0.10:
        data['originBank'] = data['originBank'].lower()
    
    # 2. Type Mismatch (Spark needs to cast String -> Float)
    elif dirt_roll < 0.20:
        data['amount'] = f"{data['amount']:,.2f}"
        
    # 3. Null Injection (Spark needs to .dropna() or .fillna())
    elif dirt_roll < 0.25:
        data['destLocation'] = None
        
    # 4. Garbage Keys (Spark needs to select specific columns)
    elif dirt_roll < 0.30:
        data['extra_garbage_field'] = "IGNORE_ME"

    return data

if __name__ == "__main__":
    print(f"🚀 Streaming Global Financial Data to: {KAFKA_TOPIC}")
    print("Normal Pattern: Trusted Country -> Trusted Country (e.g., US -> UK)")
    print("Anomaly Pattern: Trusted -> High Risk (e.g., US -> Panama), Shady Banks, or Risky Currencies")
    
    try:
        while True:
            # 1. Generate Logic
            txn = generate_transaction()
            
            # 2. Calculate Suspicion (Use clean data for logic check)
            is_suspicious = (
                txn['destBank'] in BANKS_SHADY or 
                txn['destCountry'] in COUNTRIES_HIGH_RISK or 
                txn['currency'] in CURRENCIES_RISKY or
                txn['amount'] > 100000
            )
            
            # 3. Inject Dirt (Use dirty data for Kafka)
            dirty_txn = inject_data_dirt(txn)
            
            # 4. Send
            producer.send(KAFKA_TOPIC, value=dirty_txn)
            
            # 5. Visual Log
            # We use dirty_txn for display, but is_suspicious for the icon
            display_amount = dirty_txn['amount'] 
            
            if is_suspicious:
                print(f"⚠️  ANOMALY: {dirty_txn['currency']} {display_amount:>10} | {dirty_txn['originCountry']} -> {dirty_txn['destCountry']} ({dirty_txn['destBank']})")
            else:
                print(f"✅ Normal : {dirty_txn['currency']} {display_amount:>10} | {dirty_txn['originCountry']} -> {dirty_txn['destCountry']}")
            
            time.sleep(0.5) 
            
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped.")
        producer.close()