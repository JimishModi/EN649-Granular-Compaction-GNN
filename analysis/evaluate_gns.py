"""
GNS Rollout Evaluator for Granular Compaction.
==============================================
Evaluates predicted rollout trajectories against DEM ground truth:
1. One-step position RMSE (mm)
2. Multi-step rollout position RMSE (mm)
3. Bulk density and void ratio predictions
4. Saves metrics to models/gns_results.json
"""

import os
import sys
import json
import pickle
import numpy as np

PARTICLE_RADIUS  = 0.005     # 5 mm
PARTICLE_DENSITY = 2500.0    # kg/m^3 (quartz)
BOX_WIDTH        = 0.070     # 7.0 cm die box width

def positions_to_compaction_properties(final_pos, radius=PARTICLE_RADIUS, box_w=BOX_WIDTH, rho_s=PARTICLE_DENSITY):
    n_p = final_pos.shape[0]
    final_bed_height = float(np.max(final_pos[:, 2])) + radius
    v_total = (box_w ** 2) * final_bed_height
    v_spheres = n_p * (4.0 / 3.0) * np.pi * (radius ** 3)
    v_voids = max(1e-9, v_total - v_spheres)
    mass = n_p * (4.0 / 3.0) * np.pi * (radius ** 3) * rho_s

    bulk_density = mass / v_total
    void_ratio = v_voids / v_spheres
    return float(bulk_density), float(void_ratio)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    return float(1.0 - ss_res / ss_tot)

def evaluate_gns_rollouts(rollout_dir="gns_baseline/rollouts", dataset_dir="dataset", output_path="models/gns_results.json"):
    rollout_files = sorted([
        f for f in os.listdir(rollout_dir)
        if f.endswith(".pkl") or f.endswith(".npz")
    ])

    if not rollout_files:
        raise FileNotFoundError(f"No rollout files found in {rollout_dir}")

    print(f"Found {len(rollout_files)} rollout files in {rollout_dir}")

    y_true_list = []
    y_pred_list = []
    one_step_rmse_mm = []
    full_rollout_rmse_mm = []

    for fname in rollout_files:
        fpath = os.path.join(rollout_dir, fname)

        if fname.endswith(".pkl"):
            with open(fpath, "rb") as pf:
                rollout = pickle.load(pf)
            pred_traj = np.array(rollout["predicted_rollout"])    # (T, N, 3)
            true_traj = np.array(rollout["ground_truth_rollout"]) # (T, N, 3)
        else:
            rollout = np.load(fpath, allow_pickle=True)
            pred_traj = np.array(rollout["predicted_rollout"])
            true_traj = np.array(rollout["ground_truth_rollout"])

        # Position errors
        # 1-step RMSE (first step)
        err_1step = np.sqrt(np.mean((pred_traj[0] - true_traj[0]) ** 2)) * 1000.0 # mm
        # Full rollout RMSE
        err_full = np.sqrt(np.mean((pred_traj - true_traj) ** 2)) * 1000.0 # mm
        one_step_rmse_mm.append(err_1step)
        full_rollout_rmse_mm.append(err_full)

        pred_pos = pred_traj[-1]
        true_pos = true_traj[-1]

        pred_dens, pred_void = positions_to_compaction_properties(pred_pos)
        true_dens, true_void = positions_to_compaction_properties(true_pos)

        y_pred_list.append([pred_dens, pred_void])
        y_true_list.append([true_dens, true_void])

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = {
        "one_step_rmse_mm": float(np.mean(one_step_rmse_mm)),
        "rollout_rmse_mm": float(np.mean(full_rollout_rmse_mm)),
        "density_mae": float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))),
        "density_rmse": float(np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2))),
        "density_r2": r2_score(y_true[:, 0], y_pred[:, 0]),
        "void_ratio_mae": float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))),
        "void_ratio_rmse": float(np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2))),
        "void_ratio_r2": r2_score(y_true[:, 1], y_pred[:, 1]),
    }

    results = {
        "metrics": metrics,
        "n_test_samples": len(y_true_list),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== GNS Rollout Evaluation Results ===")
    print(f"  Test samples           : {len(y_true_list)}")
    print(f"  One-step Position RMSE : {metrics['one_step_rmse_mm']:.3f} mm")
    print(f"  Full Rollout Pos RMSE  : {metrics['rollout_rmse_mm']:.3f} mm")
    print(f"  Density   R^2          : {metrics['density_r2']:.4f}")
    print(f"  Density   RMSE         : {metrics['density_rmse']:.2f} kg/m^3")
    print(f"  Void Ratio R^2         : {metrics['void_ratio_r2']:.4f}")
    print(f"  Void Ratio RMSE        : {metrics['void_ratio_rmse']:.4f}")
    print(f"  Saved to               : {output_path}")

    return results

if __name__ == "__main__":
    evaluate_gns_rollouts()
