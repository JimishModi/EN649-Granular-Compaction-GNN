"""
Unified Training Script for PyG GNN Compaction Models on RTX 4060 GPU.
Evaluates BaselineGNN and ImprovedGNN with 80/10/10 train/val/test split.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add current dir to import models and dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CompactionGraphDataset
from baseline_gnn import BaselineGNN
from improved_gnn import ImprovedGNN

def train_and_evaluate(epochs=150, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting GNN Training Pipeline on Device: {device} ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(base_dir, "dataset", "compaction_data.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found at {npz_path}. Run extract_dataset.py first!")
        
    dataset = CompactionGraphDataset(npz_path)
    num_samples = len(dataset)
    
    # 80/10/10 Train / Val / Test Split
    torch.manual_seed(42)
    indices = torch.randperm(num_samples).tolist()
    
    n_train = int(0.8 * num_samples)
    n_val = int(0.1 * num_samples)
    n_test = num_samples - n_train - n_val
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Target normalization statistics (from train set)
    train_y = torch.cat([d.y for d in train_ds], dim=0) # (N, 2)
    y_mean = train_y.mean(dim=0).to(device)
    y_std = train_y.std(dim=0).to(device) + 1e-6
    
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    results = {}
    
    for model_name, model_cls in [("BaselineGNN", BaselineGNN), ("ImprovedGNN", ImprovedGNN)]:
        print(f"\n--- Training {model_name} ---")
        model = model_cls().to(device)
        wd = 1e-3 if model_name == "ImprovedGNN" else 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=150, eta_min=5e-5
        )
        criterion = nn.MSELoss()
        
        best_val_loss     = float('inf')
        val_loss_window   = []            # smoothed over last 10 epochs
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Scale global features u to [0, 1]
                batch.u = batch.u / torch.tensor([150.0, 0.45], device=device)
                
                # Z-score normalize targets
                norm_y = (batch.y - y_mean) / y_std
                out = model(batch)
                
                loss = criterion(out, norm_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.num_graphs
                
            scheduler.step()
            train_loss = epoch_loss / len(train_ds)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    batch.u = batch.u / torch.tensor([150.0, 0.45], device=device)
                    norm_y = (batch.y - y_mean) / y_std
                    out = model(batch)
                    v_loss = criterion(out, norm_y)
                    val_loss += v_loss.item() * batch.num_graphs
            val_loss /= len(val_ds)
            val_losses.append(val_loss)
            
            # Use a smoothed window to reduce noise from tiny val set (4 samples)
            val_loss_window.append(val_loss)
            if len(val_loss_window) > 10:
                val_loss_window.pop(0)
            smoothed_val = float(np.mean(val_loss_window))
            
            if smoothed_val < best_val_loss:
                best_val_loss = smoothed_val
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_best.pth"))
                    
            if epoch % 50 == 0 or epoch == epochs:
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | SmoothedBest: {best_val_loss:.4f}")
                
        train_time = time.time() - start_time
        
        # Test Set Evaluation
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{model_name}_best.pth")))
        model.eval()
        
        y_true_list = []
        y_pred_list = []
        
        inf_start = time.time()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch.u = batch.u / torch.tensor([150.0, 0.45], device=device)
                norm_out = model(batch)
                # Un-normalize predictions
                out = norm_out * y_std + y_mean
                
                y_true_list.append(batch.y.cpu().numpy())
                y_pred_list.append(out.cpu().numpy())
                
        inf_time = (time.time() - inf_start) / len(test_ds)
        
        y_true = np.vstack(y_true_list) # (N_test, 2) -> [density, void_ratio]
        y_pred = np.vstack(y_pred_list) # (N_test, 2)
        
        # Calculate Metrics: MAE, RMSE, R^2
        mae_density = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
        rmse_density = float(np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0])**2)))
        r2_density = float(1 - np.sum((y_true[:, 0] - y_pred[:, 0])**2) / (np.sum((y_true[:, 0] - np.mean(y_true[:, 0]))**2) + 1e-8))
        
        mae_void = float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1])))
        rmse_void = float(np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1])**2)))
        r2_void = float(1 - np.sum((y_true[:, 1] - y_pred[:, 1])**2) / (np.sum((y_true[:, 1] - np.mean(y_true[:, 1]))**2) + 1e-8))
        
        results[model_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_time_sec": train_time,
            "inference_time_sec_per_sample": inf_time,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "metrics": {
                "density_mae": mae_density,
                "density_rmse": rmse_density,
                "density_r2": r2_density,
                "void_ratio_mae": mae_void,
                "void_ratio_rmse": rmse_void,
                "void_ratio_r2": r2_void
            }
        }
        
        print(f"[{model_name} Test Results] Density R²: {r2_density:.4f}, MAE: {mae_density:.2f} kg/m³ | Void Ratio R²: {r2_void:.4f}, MAE: {mae_void:.4f}")
        
    # Save training results
    res_path = os.path.join(base_dir, "models", "training_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n=== Training Complete! Results saved to {res_path} ===")
    return results

if __name__ == "__main__":
    train_and_evaluate(epochs=200, batch_size=8)
