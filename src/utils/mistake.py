import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

# Setup
fake = Faker()
NUM_RECORDS = 100000 
OUTPUT_FILE = "training_data.csv"

# --- Profiles (Must match Stream Generator) ---
COUNTRIES_TRUSTED = ['India', 'USA', 'UK', 'Germany', 'Japan', 'Singapore']
COUNTRIES_HIGH_RISK = ['Panama', 'Russia', 'Cayman Islands', 'North Korea', 'Nigeria']
CURRENCIES_TRUSTED = ['INR', 'USD', 'GBP', 'EUR', 'JPY', 'SGD']
CURRENCIES_RISKY = ['RUB', 'XMR', 'NGN', 'BTC'] 
BANKS_TRUSTED = ['HDFC', 'SBI', 'ICICI', 'Chase', 'Barclays', 'Deutsche Bank', 'HSBC', 'Mitsubishi UFJ']
BANKS_SHADY = ['Cayman Offshore', 'Panama Shell', 'Silence LLC', 'CryptoVault', 'DarkWeb Holdings']
TXN_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT']

def generate_clean_record():
    """Generates a valid (or logically anomalous) transaction record."""
    is_anomaly = random.random() < 0.05
    
    # 1. Geography & Bank Logic
    src_country = random.choice(COUNTRIES_TRUSTED)
    
    if random.random() < 0.8:
        dest_country = src_country # Domestic
    else:
        dest_country = random.choice(COUNTRIES_TRUSTED) # International Trusted
        
    src_bank = random.choice(BANKS_TRUSTED)
    dest_bank = random.choice(BANKS_TRUSTED)
    
    # Currency Logic
    currency = "USD" if src_country == "USA" else "INR" if src_country == "India" else "EUR"
    
    amount = round(random.uniform(50.0, 50000.0), 2)
    math_error = False

    # 2. Inject Logical Anomalies (The Fraud Patterns)
    if is_anomaly:
        risk_factor = random.choice(['huge_amount', 'shady_bank', 'high_risk_country', 'risky_currency', 'math_error'])
        
        if risk_factor == 'huge_amount':
            amount = round(random.uniform(200000.0, 5000000.0), 2)
        elif risk_factor == 'shady_bank':
            dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == 'high_risk_country':
            dest_country = random.choice(COUNTRIES_HIGH_RISK)
            if random.random() < 0.5: dest_bank = random.choice(BANKS_SHADY)
        elif risk_factor == 'risky_currency':
            currency = random.choice(CURRENCIES_RISKY)
        elif risk_factor == 'math_error':
            math_error = True

    # 3. Balances
    old_bal_org = round(random.uniform(amount * 1.1, 1000000.0), 2)
    new_bal_org = round(old_bal_org - amount, 2)
    old_bal_dest = round(random.uniform(0, 500000.0), 2)
    new_bal_dest = round(old_bal_dest + amount, 2)

    if math_error:
        new_bal_org = old_bal_org # Glitch

    return {
        "transactionId": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "type": random.choice(TXN_TYPES),
        "amount": amount,
        "currency": currency,
        "originBank": src_bank,
        "originLocation": f"{fake.city()}, {src_country}",
        "originCountry": src_country,
        "oldBalanceOrg": old_bal_org,
        "newBalanceOrg": new_bal_org,
        "destBank": dest_bank,
        "destLocation": f"{fake.city()}, {dest_country}",
        "destCountry": dest_country,
        "oldBalanceDest": old_bal_dest,
        "newBalanceDest": new_bal_dest,
        "is_anomaly": 1 if is_anomaly else 0 # Label for training
    }

# --- The "Dirtifier" Functions ---

def inject_nulls(df):
    """Replaces random values with NaN."""
    # 5% of destLocation becomes Null
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, 'destLocation'] = np.nan
    df.loc[mask, 'originCountry'] = np.nan
    df.loc[mask, 'timestamp'] = np.nan
    return df

def inject_casing_errors(df):
    """Mixes case: 'HDFC' -> 'hdfc'."""
    # 10% of originBank becomes lowercase
    mask = np.random.rand(len(df)) < 0.10
    df.loc[mask, 'originBank'] = df.loc[mask, 'originBank'].str.lower()
    return df

def inject_type_errors(df):
    """Converts amounts to strings with commas (e.g., '1,000.00')."""
    
    # FIX: Cast column to Object (Text) FIRST so it accepts strings
    df['amount'] = df['amount'].astype('object')
    
    # 15% of amounts become strings
    mask = np.random.rand(len(df)) < 0.15
    df.loc[mask, 'amount'] = df.loc[mask, 'amount'].apply(lambda x: f"{x:,.2f}")
    return df

def inject_duplicates(df):
    """Adds duplicate rows."""
    # Duplicate 8% of rows
    duplicates = df.sample(frac=0.08, random_state=42)
    return pd.concat([df, duplicates], ignore_index=True)

# --- Execution ---

if __name__ == "__main__":
    print(f"Step 1: Generating {NUM_RECORDS} clean global records...")
    data = [generate_clean_record() for _ in range(NUM_RECORDS)]
    df = pd.DataFrame(data)

    print("Step 2: Injecting Data Quality Issues (Dirt)...")
    df = inject_nulls(df)
    df = inject_casing_errors(df)
    df = inject_type_errors(df)
    df = inject_duplicates(df)

    # Shuffle to mix duplicates and errors
    df = df.sample(frac=1).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ SAVED: {OUTPUT_FILE}")