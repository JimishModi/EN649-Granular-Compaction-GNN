"""
Unified Cross-Validation Training & Benchmarking Engine for Granular Compaction Surrogates.
=============================================================================================
Evaluates:
1. Closed-Form Kinematic Baseline (exact rho_b and e formulas)
2. Mean Predictor Baseline (training set mean)
3. Linear Ridge Regression Baseline (on global features)
4. BaselineGNN (decoupled GNN with 64 hidden channels)
5. ImprovedGNN (physics-conditioned message passing + dual pooling with 64 hidden channels)

Targets Evaluated:
- Peak Piston Compaction Force F_peak (N)
- Instantaneous Force-Displacement Curve F(eps) (N)
- Macroscopic Bulk Density rho_b (kg/m^3) [with closed-form baseline beside it]
- Bed Void Ratio e (-) [with closed-form baseline beside it]

Protocol:
- 4-Fold GroupKFold strictly by packing seed (zero graph leakage across folds)
- Train-only feature and target normalization per fold
- Cosine Annealing learning rate schedule (T_max = epochs)
- Reports Mean +/- Std across all 4 validation folds
"""

import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath("."))
from models.dataset import CompactionGraphDataset
from models.baseline_gnn import BaselineGNN
from models.improved_gnn import ImprovedGNN

def evaluate_predictions(y_true, y_pred):
    """Computes R^2, RMSE, MAE safely."""
    # If variance is zero, r2_score is 0.0
    var_y = np.var(y_true)
    if var_y < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}

def run_cross_validation(epochs=70, batch_size=32, lr=1e-3, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting 4-Fold Grouped Cross-Validation on Device: {device} ===")
    
    # 1. Load precomputed dataset and metadata
    dataset = CompactionGraphDataset()
    with open("dataset/metadata.json", "r") as f:
        meta = json.load(f)
        
    cv_folds = meta["cv_folds"]
    print(f"Loaded {len(dataset)} total graph snapshots across {meta['num_runs']} runs.")
    
    # Storage for fold metrics
    models_to_test = ["closed_form", "mean_predictor", "linear_regression", "baseline_gnn", "improved_gnn"]
    fold_results = {m: [] for m in models_to_test}
    
    # Timing records
    timing_records = {"baseline_gnn_train_sec": [], "improved_gnn_train_sec": []}
    
    W = 0.070 # box width
    R = 0.005 # particle radius
    rho_solid = 2500.0 # kg/m^3
    v_sphere = (4.0 / 3.0) * np.pi * (R**3)
    m_particle = v_sphere * rho_solid
    
    for fold_name, fold_info in cv_folds.items():
        test_seeds = fold_info["test_seeds"]
        print(f"\n--- Processing {fold_name.upper()} (Test Seeds: {test_seeds}) ---")
        
        # Partition data into train and test graph lists
        train_graphs = [d for d in dataset if d.seed not in test_seeds]
        test_graphs = [d for d in dataset if d.seed in test_seeds]
        
        print(f"  Train: {len(train_graphs)} graphs | Test: {len(test_graphs)} graphs")
        
        # ---------------------------------------------------------
        # Train-Only Normalization Statistics
        # ---------------------------------------------------------
        y_train_all = np.array([d.y.numpy().squeeze() for d in train_graphs]) # (N_train, 3)
        y_mean = np.mean(y_train_all, axis=0) # [mean_F, mean_rho, mean_e]
        y_std = np.std(y_train_all, axis=0) + 1e-8
        
        y_mean_t = torch.tensor(y_mean, dtype=torch.float, device=device)
        y_std_t = torch.tensor(y_std, dtype=torch.float, device=device)
        
        # Ground truth test arrays
        y_test_all = np.array([d.y.numpy().squeeze() for d in test_graphs])
        test_f_true = y_test_all[:, 0]
        test_rho_true = y_test_all[:, 1]
        test_e_true = y_test_all[:, 2]
        
        # Run-level peak forces in test
        test_run_indices = sorted(list(set(d.run_idx for d in test_graphs)))
        test_peak_forces_true = []
        for r in test_run_indices:
            r_graphs = [d for d in test_graphs if d.run_idx == r]
            test_peak_forces_true.append(r_graphs[0].peak_force)
        test_peak_forces_true = np.array(test_peak_forces_true)
        
        # ---------------------------------------------------------
        # 1. Closed-Form Kinematic Baseline
        # ---------------------------------------------------------
        cf_rho_pred = []
        cf_e_pred = []
        for d in test_graphs:
            N_grains = d.num_particles
            H_set = d.settled_height
            eps = d.strain
            
            # Kinematic formulas:
            # Current bed height: H(eps) = H_set * (1 - eps)
            # Bed volume: V_bed = W^2 * H(eps)
            v_bed = (W**2) * H_set * (1.0 - eps)
            rho_calc = (N_grains * m_particle) / v_bed
            e_calc = (v_bed - N_grains * v_sphere) / (N_grains * v_sphere)
            cf_rho_pred.append(rho_calc)
            cf_e_pred.append(e_calc)
            
        cf_rho_pred = np.array(cf_rho_pred)
        cf_e_pred = np.array(cf_e_pred)
        
        fold_results["closed_form"].append({
            "rho": evaluate_predictions(test_rho_true, cf_rho_pred),
            "e": evaluate_predictions(test_e_true, cf_e_pred),
            "force": {"r2": 0.0, "rmse": 0.0, "mae": 0.0},
            "peak_force": {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
        })
        
        # ---------------------------------------------------------
        # 2. Mean Predictor Baseline
        # ---------------------------------------------------------
        mean_f_pred = np.full_like(test_f_true, y_mean[0])
        mean_rho_pred = np.full_like(test_rho_true, y_mean[1])
        mean_e_pred = np.full_like(test_e_true, y_mean[2])
        
        # Training run-level peak forces
        train_run_indices = sorted(list(set(d.run_idx for d in train_graphs)))
        train_peak_forces = [train_graphs[i*17].peak_force for i in range(len(train_run_indices))]
        mean_peak_pred = np.full_like(test_peak_forces_true, np.mean(train_peak_forces))
        
        fold_results["mean_predictor"].append({
            "force": evaluate_predictions(test_f_true, mean_f_pred),
            "rho": evaluate_predictions(test_rho_true, mean_rho_pred),
            "e": evaluate_predictions(test_e_true, mean_e_pred),
            "peak_force": evaluate_predictions(test_peak_forces_true, mean_peak_pred)
        })
        
        # ---------------------------------------------------------
        # 3. Linear Ridge Regression Baseline
        # ---------------------------------------------------------
        # Features: [gamma, eps, N, H_settled]
        X_train_lin = np.array([[d.gamma, d.strain, d.num_particles, d.settled_height] for d in train_graphs])
        X_test_lin = np.array([[d.gamma, d.strain, d.num_particles, d.settled_height] for d in test_graphs])
        
        lin_model = Ridge(alpha=1.0)
        lin_model.fit(X_train_lin, y_train_all)
        lin_preds = lin_model.predict(X_test_lin)
        
        # Run-level peak force ridge regression: features [gamma, N, H_settled]
        X_train_peak = np.array([[train_graphs[i*17].gamma, train_graphs[i*17].num_particles, train_graphs[i*17].settled_height] for i in range(len(train_run_indices))])
        X_test_peak = np.array([[r_graphs[0].gamma, r_graphs[0].num_particles, r_graphs[0].settled_height] for r in test_run_indices for r_graphs in [[d for d in test_graphs if d.run_idx == r]]])
        lin_peak_model = Ridge(alpha=1.0)
        lin_peak_model.fit(X_train_peak, train_peak_forces)
        lin_peak_preds = lin_peak_model.predict(X_test_peak)
        
        fold_results["linear_regression"].append({
            "force": evaluate_predictions(test_f_true, lin_preds[:, 0]),
            "rho": evaluate_predictions(test_rho_true, lin_preds[:, 1]),
            "e": evaluate_predictions(test_e_true, lin_preds[:, 2]),
            "peak_force": evaluate_predictions(test_peak_forces_true, lin_peak_preds)
        })
        
        # ---------------------------------------------------------
        # 4. BaselineGNN Training
        # ---------------------------------------------------------
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        model_base = BaselineGNN(in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3).to(device)
        optimizer_base = torch.optim.Adam(model_base.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, T_max=epochs)
        criterion = nn.MSELoss()
        
        t0_base = time.time()
        for epoch in range(epochs):
            model_base.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer_base.zero_grad()
                # Target normalized by training statistics
                y_norm = (batch.y - y_mean_t) / y_std_t
                pred = model_base(batch)
                loss = criterion(pred, y_norm)
                loss.backward()
                optimizer_base.step()
            scheduler_base.step()
        timing_records["baseline_gnn_train_sec"].append(time.time() - t0_base)
        
        # Evaluate BaselineGNN on test graphs
        model_base.eval()
        base_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_norm = model_base(batch)
                pred_orig = pred_norm * y_std_t + y_mean_t
                base_preds.append(pred_orig.cpu().numpy())
        base_preds = np.concatenate(base_preds, axis=0)
        
        # Compute run-level peak forces: max over strain snapshots
        base_peak_preds = []
        for r in test_run_indices:
            mask = [d.run_idx == r for d in test_graphs]
            r_preds_f = base_preds[mask, 0]
            base_peak_preds.append(np.max(r_preds_f))
        base_peak_preds = np.array(base_peak_preds)
        
        fold_results["baseline_gnn"].append({
            "force": evaluate_predictions(test_f_true, base_preds[:, 0]),
            "rho": evaluate_predictions(test_rho_true, base_preds[:, 1]),
            "e": evaluate_predictions(test_e_true, base_preds[:, 2]),
            "peak_force": evaluate_predictions(test_peak_forces_true, base_peak_preds)
        })
        
        # ---------------------------------------------------------
        # 5. ImprovedGNN Training
        # ---------------------------------------------------------
        model_imp = ImprovedGNN(in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3).to(device)
        optimizer_imp = torch.optim.Adam(model_imp.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_imp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_imp, T_max=epochs)
        
        t0_imp = time.time()
        for epoch in range(epochs):
            model_imp.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer_imp.zero_grad()
                y_norm = (batch.y - y_mean_t) / y_std_t
                pred = model_imp(batch)
                loss = criterion(pred, y_norm)
                loss.backward()
                optimizer_imp.step()
            scheduler_imp.step()
        timing_records["improved_gnn_train_sec"].append(time.time() - t0_imp)
        
        # Evaluate ImprovedGNN on test graphs
        model_imp.eval()
        imp_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_norm = model_imp(batch)
                pred_orig = pred_norm * y_std_t + y_mean_t
                imp_preds.append(pred_orig.cpu().numpy())
        imp_preds = np.concatenate(imp_preds, axis=0)
        
        # Compute run-level peak forces
        imp_peak_preds = []
        for r in test_run_indices:
            mask = [d.run_idx == r for d in test_graphs]
            r_preds_f = imp_preds[mask, 0]
            imp_peak_preds.append(np.max(r_preds_f))
        imp_peak_preds = np.array(imp_peak_preds)
        
        fold_results["improved_gnn"].append({
            "force": evaluate_predictions(test_f_true, imp_preds[:, 0]),
            "rho": evaluate_predictions(test_rho_true, imp_preds[:, 1]),
            "e": evaluate_predictions(test_e_true, imp_preds[:, 2]),
            "peak_force": evaluate_predictions(test_peak_forces_true, imp_peak_preds)
        })
        
        print(f"  [Fold Complete] BaselineGNN Force R^2: {fold_results['baseline_gnn'][-1]['force']['r2']:.4f} | ImprovedGNN Force R^2: {fold_results['improved_gnn'][-1]['force']['r2']:.4f}")
        print(f"                  BaselineGNN Peak F R^2: {fold_results['baseline_gnn'][-1]['peak_force']['r2']:.4f} | ImprovedGNN Peak F R^2: {fold_results['improved_gnn'][-1]['peak_force']['r2']:.4f}")
        print(f"                  Closed-Form rho R^2: {fold_results['closed_form'][-1]['rho']['r2']:.4f} | Closed-Form e R^2: {fold_results['closed_form'][-1]['e']['r2']:.4f}")

    # -------------------------------------------------------------
    # Aggregate Metrics Across All 4 Folds
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4-FOLD GROUPED CROSS-VALIDATION FINAL BENCHMARK RESULTS")
    print("=" * 80)
    
    summary = {}
    for model_name in models_to_test:
        summary[model_name] = {}
        for target in ["peak_force", "force", "rho", "e"]:
            r2_vals = [fold_results[model_name][f][target]["r2"] for f in range(4)]
            rmse_vals = [fold_results[model_name][f][target]["rmse"] for f in range(4)]
            mae_vals = [fold_results[model_name][f][target]["mae"] for f in range(4)]
            summary[model_name][target] = {
                "r2_mean": float(np.mean(r2_vals)),
                "r2_std": float(np.std(r2_vals)),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "mae_mean": float(np.mean(mae_vals)),
                "mae_std": float(np.std(mae_vals))
            }
            
    # Save results to json
    os.makedirs("models", exist_ok=True)
    out_json = {
        "summary": summary,
        "fold_results": fold_results,
        "timing_records": timing_records
    }
    with open("models/cv_benchmark_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("Saved benchmark results to models/cv_benchmark_results.json")
    
    # Print formatted markdown table
    print("\n### Cross-Validation Performance Summary Table (Mean +/- Std Across 4 Folds)\n")
    print("| Model Architecture | Peak Force F_peak R^2 | F(eps) Curve R^2 | Bulk Density rho_b R^2 | Void Ratio e R^2 |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    def fmt_cell(m_dict, target):
        r2_m = m_dict[target]["r2_mean"]
        r2_s = m_dict[target]["r2_std"]
        return f"${r2_m:.4f} \\pm {r2_s:.4f}$"
        
    print(f"| **Closed-Form Formula** | *N/A (kinematic)* | *N/A (kinematic)* | **{fmt_cell(summary['closed_form'], 'rho')}** | **{fmt_cell(summary['closed_form'], 'e')}** |")
    print(f"| **Mean Predictor** | {fmt_cell(summary['mean_predictor'], 'peak_force')} | {fmt_cell(summary['mean_predictor'], 'force')} | {fmt_cell(summary['mean_predictor'], 'rho')} | {fmt_cell(summary['mean_predictor'], 'e')} |")
    print(f"| **Linear Ridge Regression** | {fmt_cell(summary['linear_regression'], 'peak_force')} | {fmt_cell(summary['linear_regression'], 'force')} | {fmt_cell(summary['linear_regression'], 'rho')} | {fmt_cell(summary['linear_regression'], 'e')} |")
    print(f"| **BaselineGNN** (64-dim) | {fmt_cell(summary['baseline_gnn'], 'peak_force')} | {fmt_cell(summary['baseline_gnn'], 'force')} | {fmt_cell(summary['baseline_gnn'], 'rho')} | {fmt_cell(summary['baseline_gnn'], 'e')} |")
    print(f"| **ImprovedGNN** (64-dim, u-injected) | **{fmt_cell(summary['improved_gnn'], 'peak_force')}** | **{fmt_cell(summary['improved_gnn'], 'force')}** | {fmt_cell(summary['improved_gnn'], 'rho')} | {fmt_cell(summary['improved_gnn'], 'e')} |")
    
    # Save the trained ImprovedGNN model weights for downstream evaluation
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model_imp.state_dict(), "models/checkpoints/ImprovedGNN_best.pth")
    torch.save(model_base.state_dict(), "models/checkpoints/BaselineGNN_best.pth")
    print("\nSaved best model checkpoints to models/checkpoints/")
    return summary

if __name__ == "__main__":
    run_cross_validation(epochs=70, batch_size=32, lr=1e-3, weight_decay=1e-4)
