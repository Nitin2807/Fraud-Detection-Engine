import torch
import torch.optim as optim
import torch.nn as nn
import os
import joblib
# Import your model class and data prep
from src.training.model import Autoencoder, DEVICE
from src.processing.preprocessed import preprocess_data

# --- CONFIG ---
DATA_PATH = "training_data.csv"
MODEL_SAVE_DIR = "models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. THE WINNING PARAMS (From your message) ---
best_params = {
    'num_layers': 1, 
    'start_neurons': 96, 
    'dropout': 0.1, 
    'lr': 0.0007624212166378842, 
    'batch_size': 128, 
    'epochs': 25
}

# --- 2. RETRAIN (Just once, with the winner) ---
print("🏆 Retraining the Best Model to save it...")

# Load Data
TRAIN_TENSOR = preprocess_data(DATA_PATH, MODEL_SAVE_DIR).to(DEVICE)
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(TRAIN_TENSOR)
loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

# Build Model
input_dim = TRAIN_TENSOR.shape[1]
model = Autoencoder(
    input_dim=input_dim, 
    output_neurons_layer=best_params['start_neurons'], 
    dropout_rate=best_params['dropout'], 
    num_layers=best_params['num_layers']
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

# Train Loop
model.train()
for epoch in range(best_params['epochs']):
    total_loss = 0
    for batch in loader:
        batch_data = batch[0].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{best_params['epochs']} - Loss: {total_loss/len(loader):.5f}")

# --- 3. SAVE IT ---
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_autoencoder.pth"))
print(f"✅ Saved correctly to: {MODEL_SAVE_DIR}")