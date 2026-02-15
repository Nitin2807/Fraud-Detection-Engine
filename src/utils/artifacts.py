import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import your modules
from src.training.model import Autoencoder, DEVICE
# We CANNOT use the standard preprocess function because it returns a Tensor.
# We need to recreate the Scaler logic manually here to save it.

# --- CONFIG ---
DATA_PATH = "training_data.csv"
MODEL_SAVE_DIR = "models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. RECREATE THE SCALER ---
print("🔄 Regenerating Standard Scaler...")
df = pd.read_csv(DATA_PATH)

# Clean & Prep (Must match training exactly!)
if df['amount'].dtype == 'object':
    df['amount'] = df['amount'].str.replace(',', '').astype(float)
df['originBank'] = df['originBank'].str.upper()
df.fillna("UNKNOWN", inplace=True)

# Drop unused columns
drop_columns = ['transactionId', 'timestamp', 'originLocation', 'destLocation', 'is_anomaly']
df_features = df.drop(columns=drop_columns, errors='ignore')

# Encode
cat_cols = ['type', 'currency', 'originBank', 'originCountry', 'destBank', 'destCountry']
num_cols = ['amount', 'oldBalanceOrg', 'newBalanceOrg', 'oldBalanceDest', 'newBalanceDest']
df_encoded = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)

# SAVE COLUMN NAMES (Crucial for Spark to match schema)
joblib.dump(df_encoded.columns, os.path.join(MODEL_SAVE_DIR, "model_columns.bin"))
print(f"✅ Column names saved ({len(df_encoded.columns)} features)")

# Fit & Save Scaler
scaler = StandardScaler()
scaler.fit(df_encoded[num_cols]) # Fit only on numericals
joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "std_scaler.bin"))
print("✅ Scaler saved successfully.")

# --- 2. RESAVE THE MODEL (Using Best Params) ---
# Your best params from Optuna
best_params = {
    'num_layers': 1, 
    'start_neurons': 96, 
    'dropout': 0.1, 
    'lr': 0.0007624, 
    'batch_size': 128, 
    'epochs': 25
}

print("🏆 Re-initializing Best Model...")
input_dim = len(df_encoded.columns)

model = Autoencoder(
    input_dim=input_dim, 
    output_neurons_layer=best_params['start_neurons'], 
    dropout_rate=best_params['dropout'], 
    num_layers=best_params['num_layers']
).to(DEVICE)

# Save the initialized model structure (weights will be random until trained, 
# but if you have the .pth from the run, overwrite this!)
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_autoencoder.pth"))
print(f"✅ Model structure saved to {MODEL_SAVE_DIR}")