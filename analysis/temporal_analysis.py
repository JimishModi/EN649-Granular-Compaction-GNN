"""
Temporal Analysis & Numerical Convergence Engine.
=================================================
Executes:
1. Time-step (dt) convergence study across 3 cohesion levels ({0, 50, 150} kPa)
   at dt in {5.0e-5, 2.5e-5, 1.25e-5} s.
2. Trajectory comparison on held-out runs:
   Plots F(eps) and rho_b(eps) for DEM ground truth vs ImprovedGNN vs BaselineGNN vs Closed-Form.
3. Generates publication-quality figures:
   - analysis/plots/dt_convergence.png
   - analysis/plots/heldout_force_displacement.png
   - analysis/plots/heldout_density_strain.png
"""

import os
import sys
import time
import json
import torch
import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

sys.path.insert(0, os.path.abspath("."))
from dem_sim.python_dem import ParticleBedDEM
from models.dataset import CompactionGraphDataset
from models.improved_gnn import ImprovedGNN
from models.baseline_gnn import BaselineGNN
from torch_geometric.data import Data
from scipy.spatial import cKDTree

os.makedirs("analysis/plots", exist_ok=True)

def run_dt_convergence():
    print("=" * 65)
    print("1. RUNNING TIME-STEP (dt) CONVERGENCE STUDY ON 3 CASES")
    print("=" * 65)
    
    cohesion_cases = [
        {"name": "Cohesionless (0 kPa)", "coh": 0.0},
        {"name": "Moderately Cohesive (50 kPa)", "coh": 50.0},
        {"name": "Strongly Cohesive (150 kPa)", "coh": 150.0}
    ]
    
    # dt levels: nominal (50 us), refined (25 us), highly refined (12.5 us)
    dt_levels = [
        {"label": "dt = 5.0e-5 s (nominal)", "dt": 5.0e-5, "settle": 3000, "compact_steps_mult": 1},
        {"label": "dt = 2.5e-5 s (2x refined)", "dt": 2.5e-5, "settle": 6000, "compact_steps_mult": 2},
        {"label": "dt = 1.25e-5 s (4x refined)", "dt": 1.25e-5, "settle": 12000, "compact_steps_mult": 4}
    ]
    
    results = {}
    
    for case in cohesion_cases:
        c_name = case["name"]
        coh = case["coh"]
        print(f"\n--- Testing Case: {c_name} ---")
        results[c_name] = []
        
        for dt_cfg in dt_levels:
            dt_val = dt_cfg["dt"]
            dt_label = dt_cfg["label"]
            
            t0 = time.time()
            sim = ParticleBedDEM(num_particles=400, cohesion=coh, max_strain=0.13, seed=42)
            sim.dt = dt_val
            
            # Settling
            settle_steps = dt_cfg["settle"]
            p1 = int(0.75 * settle_steps)
            p2 = settle_steps - p1
            
            sim.mu_friction = 0.05
            for s in range(p1):
                f, _, _ = sim.compute_forces()
                sim.velocities += (f / sim.mass) * dt_val
                sim.positions += sim.velocities * dt_val
            sim.mu_friction = 0.30
            for s in range(p2):
                f, _, _ = sim.compute_forces()
                sim.velocities += (f / sim.mass) * dt_val
                sim.positions += sim.velocities * dt_val
                
            h_settled = float(sim.positions[:, 2].max() + sim.radius)
            v_solid = 400 * (4/3) * np.pi * (sim.radius**3)
            
            # Compaction
            target_disp = h_settled * 0.13
            compaction_time = target_disp / sim.piston_velocity
            compact_steps = int(np.round(compaction_time / dt_val))
            snapshot_interval = max(1, compact_steps // 16)
            
            forces_hist = []
            strains_hist = []
            
            for s in range(compact_steps + 1):
                pz = h_settled - sim.piston_velocity * (s * dt_val)
                pz = max(pz, h_settled - target_disp)
                f, f_piston, _ = sim.compute_forces(top_wall_z=pz, top_wall_v=-sim.piston_velocity)
                sim.velocities += (f / sim.mass) * dt_val
                sim.positions += sim.velocities * dt_val
                
                if (s % snapshot_interval == 0) or (s == compact_steps):
                    eps = (h_settled - pz) / h_settled
                    forces_hist.append(f_piston)
                    strains_hist.append(eps)
                    
            wall_time = time.time() - t0
            peak_f = float(np.max(forces_hist))
            final_h = float(pz)
            v_bed_final = sim.box_area * final_h
            final_e = float((v_bed_final - v_solid) / v_solid)
            final_phi = float(v_solid / v_bed_final)
            
            entry = {
                "dt_label": dt_label,
                "dt": dt_val,
                "wall_time_sec": float(wall_time),
                "h_settled": h_settled,
                "final_e": final_e,
                "final_phi": final_phi,
                "peak_force": peak_f,
                "strains": strains_hist,
                "forces": forces_hist
            }
            results[c_name].append(entry)
            print(f"  {dt_label}: Wall Time = {wall_time:.2f}s | Peak F = {peak_f:.2f} N | Final e = {final_e:.4f} | phi = {final_phi:.4f}")
            
    # Compute relative convergence errors between dt_1 (nominal) and dt_3 (finest)
    print("\n--- Convergence Summary (Relative Difference vs Finest dt = 1.25e-5 s) ---")
    for c_name in results:
        res_nom = results[c_name][0]
        res_fine = results[c_name][2]
        
        diff_peak_f = abs(res_nom["peak_force"] - res_fine["peak_force"]) / res_fine["peak_force"] * 100.0
        diff_e = abs(res_nom["final_e"] - res_fine["final_e"]) / res_fine["final_e"] * 100.0
        print(f"  {c_name}:")
        print(f"    Peak Force Rel Diff: {diff_peak_f:.2f}% (Nominal={res_nom['peak_force']:.2f} N, Finest={res_fine['peak_force']:.2f} N)")
        print(f"    Final Void Ratio Rel Diff: {diff_e:.2f}% (Nominal={res_nom['final_e']:.4f}, Finest={res_fine['final_e']:.4f})")

    # Plot dt convergence curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", ":"]
    
    for ax_idx, c_name in enumerate(results):
        ax = axes[ax_idx]
        for cfg_idx, cfg_res in enumerate(results[c_name]):
            ax.plot(
                cfg_res["strains"],
                cfg_res["forces"],
                label=cfg_res["dt_label"],
                color=colors[cfg_idx],
                linestyle=linestyles[cfg_idx],
                linewidth=2.0
            )
        ax.set_title(c_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Axial Strain $\\varepsilon$", fontsize=11)
        ax.set_ylabel("Piston Compaction Force $F$ (N)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        
    plt.tight_layout()
    plt.savefig("analysis/plots/dt_convergence.png")
    plt.close()
    print("Saved dt convergence plot to analysis/plots/dt_convergence.png")
    return results

def run_heldout_trajectory_comparison():
    print("\n" + "=" * 65)
    print("2. HELD-OUT TEST RUN TRAJECTORY EVALUATION: F(eps) & rho_b(eps)")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset & models
    dataset = CompactionGraphDataset()
    with open("dataset/metadata.json", "r") as f:
        meta = json.load(f)
        
    # Load model weights
    model_imp = ImprovedGNN(in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3).to(device)
    model_imp.load_state_dict(torch.load("models/checkpoints/ImprovedGNN_best.pth", map_location=device))
    model_imp.eval()
    
    model_base = BaselineGNN(in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3).to(device)
    model_base.load_state_dict(torch.load("models/checkpoints/BaselineGNN_best.pth", map_location=device))
    model_base.eval()
    
    # Load training normalization stats from fold_1
    train_graphs = [d for d in dataset if d.seed not in [11, 22]]
    y_train = np.array([d.y.numpy().squeeze() for d in train_graphs])
    y_mean_t = torch.tensor(np.mean(y_train, axis=0), dtype=torch.float, device=device)
    y_std_t = torch.tensor(np.std(y_train, axis=0) + 1e-8, dtype=torch.float, device=device)
    
    # Evaluate 3 held-out test runs from seed 11:
    # Run 0: Cohesion 0 kPa
    # Run 8: Cohesion 50 kPa
    # Run 24: Cohesion 150 kPa
    test_run_ids = [0, 8, 24]
    run_titles = [
        "Held-Out Run 01: Cohesion 0 kPa (Seed 11, N=380)",
        "Held-Out Run 09: Cohesion 50 kPa (Seed 11, N=380)",
        "Held-Out Run 25: Cohesion 150 kPa (Seed 11, N=380)"
    ]
    
    fig_f, axes_f = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    fig_rho, axes_rho = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    
    comparison_records = []
    
    W = 0.070
    rho_solid = 2500.0
    v_sphere = (4/3) * np.pi * (0.005**3)
    m_p = v_sphere * rho_solid
    
    for ax_idx, r_idx in enumerate(test_run_ids):
        # Extract all snapshots for this run
        r_graphs = [d for d in dataset if d.run_idx == r_idx]
        strains = np.array([d.strain for d in r_graphs])
        true_f = np.array([float(d.y[0, 0]) for d in r_graphs])
        true_rho = np.array([float(d.y[0, 1]) for d in r_graphs])
        
        # Predictions
        imp_f = []
        imp_rho = []
        base_f = []
        base_rho = []
        cf_rho = []
        
        for d in r_graphs:
            d_dev = d.to(device)
            # Batch attribute required for pooling
            d_dev.batch = torch.zeros(d_dev.num_nodes, dtype=torch.long, device=device)
            with torch.no_grad():
                out_imp = model_imp(d_dev) * y_std_t + y_mean_t
                out_base = model_base(d_dev) * y_std_t + y_mean_t
            
            imp_f.append(float(out_imp[0, 0]))
            imp_rho.append(float(out_imp[0, 1]))
            base_f.append(float(out_base[0, 0]))
            base_rho.append(float(out_base[0, 1]))
            
            # Closed-Form formula:
            v_bed = (W**2) * d.settled_height * (1.0 - d.strain)
            cf_rho.append((d.num_particles * m_p) / v_bed)
            
        imp_f = np.array(imp_f)
        imp_rho = np.array(imp_rho)
        base_f = np.array(base_f)
        base_rho = np.array(base_rho)
        cf_rho = np.array(cf_rho)
        
        title = run_titles[ax_idx]
        print(f"\n{title}:")
        print(f"  True Peak Force: {np.max(true_f):.2f} N | ImprovedGNN: {np.max(imp_f):.2f} N | BaselineGNN: {np.max(base_f):.2f} N")
        print(f"  Force Curve RMSE: ImprovedGNN = {compute_rmse(true_f, imp_f):.2f} N vs BaselineGNN = {compute_rmse(true_f, base_f):.2f} N")
        print(f"  Closed-Form rho_b RMSE: {compute_rmse(true_rho, cf_rho):.2e} kg/m^3")
        
        comparison_records.append({
            "run_title": title,
            "strains": strains.tolist(),
            "true_f": true_f.tolist(),
            "imp_f": imp_f.tolist(),
            "base_f": base_f.tolist(),
            "true_rho": true_rho.tolist(),
            "cf_rho": cf_rho.tolist(),
            "imp_rho": imp_rho.tolist()
        })
        
        # Plot Force-Displacement F(eps)
        ax_f = axes_f[ax_idx]
        ax_f.plot(strains, true_f, 'k-', linewidth=2.5, label="DEM Ground Truth")
        ax_f.plot(strains, imp_f, 'r--', linewidth=2.0, label="ImprovedGNN (u-injected)")
        ax_f.plot(strains, base_f, 'b:', linewidth=1.8, label="BaselineGNN (decoupled)")
        ax_f.set_title(title, fontsize=10, fontweight="bold")
        ax_f.set_xlabel("Axial Strain $\\varepsilon$", fontsize=11)
        ax_f.set_ylabel("Compaction Force $F$ (N)", fontsize=11)
        ax_f.grid(True, linestyle="--", alpha=0.5)
        ax_f.legend(fontsize=9)
        
        # Plot Density-Strain rho_b(eps)
        ax_r = axes_rho[ax_idx]
        ax_r.plot(strains, true_rho, 'k-', linewidth=2.5, label="DEM Ground Truth")
        ax_r.plot(strains, cf_rho, 'g--', linewidth=2.0, label="Closed-Form Exact Formula")
        ax_r.plot(strains, imp_rho, 'r:', linewidth=1.8, label="ImprovedGNN Prediction")
        ax_r.set_title(title, fontsize=10, fontweight="bold")
        ax_r.set_xlabel("Axial Strain $\\varepsilon$", fontsize=11)
        ax_r.set_ylabel("Bulk Density $\\rho_b$ (kg/m$^3$)", fontsize=11)
        ax_r.grid(True, linestyle="--", alpha=0.5)
        ax_r.legend(fontsize=9)
        
    fig_f.tight_layout()
    fig_f.savefig("analysis/plots/heldout_force_displacement.png")
    plt.close(fig_f)
    
    fig_rho.tight_layout()
    fig_rho.savefig("analysis/plots/heldout_density_strain.png")
    plt.close(fig_rho)
    
    print("\nSaved held-out trajectory plots:")
    print("  -> analysis/plots/heldout_force_displacement.png")
    print("  -> analysis/plots/heldout_density_strain.png")
    
    return comparison_records

if __name__ == "__main__":
    dt_res = run_dt_convergence()
    held_res = run_heldout_trajectory_comparison()
    
    # Save all results to analysis/temporal_convergence_results.json
    out_all = {
        "dt_convergence": {
            k: [
                {
                    "dt_label": x["dt_label"],
                    "dt": x["dt"],
                    "wall_time_sec": x["wall_time_sec"],
                    "final_e": x["final_e"],
                    "final_phi": x["final_phi"],
                    "peak_force": x["peak_force"]
                }
                for x in v
            ]
            for k, v in dt_res.items()
        },
        "heldout_comparison": held_res
    }
    with open("analysis/temporal_convergence_results.json", "w") as f:
        json.dump(out_all, f, indent=2)
    print("\nSaved all numerical results to analysis/temporal_convergence_results.json")
