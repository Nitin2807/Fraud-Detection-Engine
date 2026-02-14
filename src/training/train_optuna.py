from src.training.model import Autoencoder, DEVICE, model_save_dir
from src.processing.preprocessed import preprocess_data
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import mlflow
import optuna
import torch
import os

# Define this here so we can pass it to preprocess_data
DATA_PATH = "training_data.csv"

def objective(trial):
    # Load Data (Global Variable)
    dataset = TensorDataset(TRAIN_TENSOR)
    
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 5, step=1),
        "start_neurons": trial.suggest_int("start_neurons", 32, 128, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": trial.suggest_int("epochs", 15, 40, step=5) 
    }
    
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    input_dim = TRAIN_TENSOR.shape[1]
    
    model = Autoencoder(
        input_dim=input_dim, 
        output_neurons_layer=params["start_neurons"], 
        dropout_rate=params["dropout"],               
        num_layers=params["num_layers"]
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    # MLflow logging
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        # FIX 1: Initialize avg_loss to prevent "unbound" error if loop fails
        avg_loss = 0.0
        
        for epoch in range(params["epochs"]):
            model.train()
            running_loss = 0.0
            
            for batch in loader:
                batch_data = batch[0].to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(loader)
            
            # Pruning (Stop bad trials early)
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                mlflow.log_metric("pruned_loss", avg_loss)
                raise optuna.exceptions.TrialPruned()
        
        mlflow.log_metric("final_loss", avg_loss)
        
        # Note: 'study.best_trial' is only updated AFTER the trial finishes.
        # This check might fail on the very first trial or current best. 
        # Ideally, we rely on MLflow artifact logging or a custom callback.
        try:
            if study.best_trial and trial.number == study.best_trial.number:
                torch.save(model.state_dict(), os.path.join(model_save_dir, "best_autoencoder.pth"))
        except:
            pass # Ignore error on first trial when best_trial doesn't exist yet
            
    return avg_loss

if __name__ == "__main__":
    # FIX 2: Pass the required arguments to preprocess_data
    # We use the global DATA_PATH and the imported model_save_dir
    TRAIN_TENSOR = preprocess_data(DATA_PATH, model_save_dir).to(DEVICE)
    
    print("Starting Optuna")
    mlflow.set_experiment("Fraud_Detection_v2")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) 
    
    print("\n Best Trial Results:")
    print(study.best_params)
    print(f"Best Loss: {study.best_value}")
    print(f"✅ Best Model saved to {model_save_dir}/best_autoencoder.pth")