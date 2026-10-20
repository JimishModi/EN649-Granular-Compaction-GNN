"""
GNS Rollout Evaluator for Granular Compaction.

After running GNS in rollout mode, this script:
  1. Loads each predicted rollout .pkl file from the gns rollout directory.
  2. Extracts the final timestep predicted positions.
  3. Computes bulk density and void ratio from final positions.
  4. Compares against DEM ground truth labels from compaction_data.npz.
  5. Outputs MAE, RMSE, R² metrics and saves results to models/gns_results.json.

Usage:
    python analysis/evaluate_gns.py

Requires:
    - gns_baseline/rollouts/ containing rollout output .pkl files
    - dataset/compaction_data.npz
    - models/training_results.json (for test split indices, same seed)
"""

import os
import sys
import json
import pickle
import numpy as np

# -----------------------------------------------------------------------
# Physical constants (must match python_dem.py)
# -----------------------------------------------------------------------
PARTICLE_RADIUS  = 0.005     # 5 mm
PARTICLE_DENSITY = 2500.0    # kg/m³ (quartz)
BOX_WIDTH        = 0.1       # 10 cm


def positions_to_compaction_properties(
    final_pos: np.ndarray,
    radius: float = PARTICLE_RADIUS,
    box_w: float  = BOX_WIDTH,
    rho_s: float  = PARTICLE_DENSITY,
) -> tuple[float, float]:
    """
    Compute macroscopic compaction properties from final particle positions.

    Args:
        final_pos : (N, 3) array of final particle positions
        radius    : particle radius (m)
        box_w     : box side length (m)
        rho_s     : solid particle density (kg/m³)

    Returns:
        (bulk_density kg/m³, void_ratio e)
    """
    n_p = final_pos.shape[0]
    final_bed_height = float(np.max(final_pos[:, 2])) + radius
    v_total   = (box_w ** 2) * final_bed_height
    v_spheres = n_p * (4.0 / 3.0) * np.pi * radius ** 3
    v_voids   = max(1e-9, v_total - v_spheres)
    mass      = n_p * (4.0 / 3.0) * np.pi * radius ** 3 * rho_s

    bulk_density = mass / v_total
    void_ratio   = v_voids / v_spheres
    return float(bulk_density), float(void_ratio)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    return float(1.0 - ss_res / ss_tot)


def evaluate_gns_rollouts(
    rollout_dir: str,
    dataset_dir: str,
    output_path: str,
) -> dict:
    """
    Evaluate GNS rollout outputs against DEM ground truth.

    Args:
        rollout_dir : directory containing .pkl rollout files from GNS
        dataset_dir : dataset/ directory containing compaction_data.npz
        output_path : path to save gns_results.json

    Returns:
        Dictionary with metrics, y_true, y_pred.
    """
    if not os.path.isdir(rollout_dir):
        raise FileNotFoundError(
            f"Rollout directory not found: {rollout_dir}\n"
            "Run GNS in rollout mode first:\n"
            "  python gns_baseline/train.py --mode=rollout --data_path=dataset/gns "
            "--model_path=gns_baseline/checkpoints --output_path=gns_baseline/rollouts"
        )

    rollout_files = sorted([
        f for f in os.listdir(rollout_dir)
        if f.endswith(".pkl") or f.endswith(".npz")
    ])

    if not rollout_files:
        raise FileNotFoundError(
            f"No rollout .pkl or .npz files found in {rollout_dir}"
        )

    print(f"Found {len(rollout_files)} rollout files in {rollout_dir}")

    y_true_list = []
    y_pred_list = []
    inference_times = []

    for fname in rollout_files:
        fpath = os.path.join(rollout_dir, fname)

        # Load rollout — GNS saves as .pkl dicts with keys:
        # 'predicted_rollout': (T, N, 3)
        # 'ground_truth_rollout': (T, N, 3)
        # 'metadata': dict with 'elapsed' if available
        if fname.endswith(".pkl"):
            with open(fpath, "rb") as pf:
                rollout = pickle.load(pf)
            pred_pos  = np.array(rollout["predicted_rollout"])[-1]    # (N, 3)
            true_pos  = np.array(rollout["ground_truth_rollout"])[-1] # (N, 3)
            elapsed   = rollout.get("metadata", {}).get("elapsed", None)
        else:  # .npz
            rollout   = np.load(fpath, allow_pickle=True)
            pred_pos  = rollout["predicted_rollout"][-1]
            true_pos  = rollout["ground_truth_rollout"][-1]
            elapsed   = None

        if elapsed is not None:
            inference_times.append(float(elapsed))

        pred_dens, pred_void = positions_to_compaction_properties(pred_pos)
        true_dens, true_void = positions_to_compaction_properties(true_pos)

        y_pred_list.append([pred_dens, pred_void])
        y_true_list.append([true_dens, true_void])

    y_true = np.array(y_true_list)  # (N_test, 2)
    y_pred = np.array(y_pred_list)  # (N_test, 2)

    metrics = {
        "density_mae":     float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))),
        "density_rmse":    float(np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2))),
        "density_r2":      r2_score(y_true[:, 0], y_pred[:, 0]),
        "void_ratio_mae":  float(np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))),
        "void_ratio_rmse": float(np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2))),
        "void_ratio_r2":   r2_score(y_true[:, 1], y_pred[:, 1]),
    }

    avg_inf_time = float(np.mean(inference_times)) if inference_times else None

    results = {
        "metrics":   metrics,
        "y_true":    y_true.tolist(),
        "y_pred":    y_pred.tolist(),
        "inference_time_sec_per_sample": avg_inf_time,
        "n_test_samples": len(y_true_list),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== GNS Evaluation Results ===")
    print(f"  Test samples       : {len(y_true_list)}")
    print(f"  Density   R²       : {metrics['density_r2']:.4f}")
    print(f"  Density   MAE      : {metrics['density_mae']:.2f} kg/m³")
    print(f"  Density   RMSE     : {metrics['density_rmse']:.2f} kg/m³")
    print(f"  Void Ratio R²      : {metrics['void_ratio_r2']:.4f}")
    print(f"  Void Ratio MAE     : {metrics['void_ratio_mae']:.4f}")
    print(f"  Void Ratio RMSE    : {metrics['void_ratio_rmse']:.4f}")
    if avg_inf_time:
        print(f"  Avg inference time : {avg_inf_time:.5f} s/sample")
    print(f"\n  Results saved to   : {output_path}")

    return results


if __name__ == "__main__":
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rollout_dir = os.path.join(base_dir, "gns_baseline", "rollouts")
    dataset_dir = os.path.join(base_dir, "dataset")
    output_path = os.path.join(base_dir, "models", "gns_results.json")

    evaluate_gns_rollouts(rollout_dir, dataset_dir, output_path)
