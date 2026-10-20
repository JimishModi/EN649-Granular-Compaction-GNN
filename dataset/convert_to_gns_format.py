"""
GNS Dataset Converter for Granular Compaction Data.

Converts compaction_data.npz into the format expected by geoelements/gns:
  dataset/gns/
    ├── metadata.json   (GNS-format: bounds, connectivity_radius, normalization stats)
    ├── train.npz       (80% of runs)
    ├── valid.npz       (10% of runs)
    └── test.npz        (10% of runs)

Each .npz contains:
  - 'pos'           : object array of shape (n_traj,), each entry (T, n_particles, 3)
  - 'particle_type' : object array of shape (n_traj,), each entry (n_particles,) int

Run after extract_dataset.py has been executed.
"""

import os
import json
import numpy as np


def convert_to_gns_format(dataset_dir: str) -> str:
    """
    Reads compaction_data.npz + metadata.json, writes GNS-compatible splits.

    Returns:
        Path to the created gns/ subdirectory.
    """
    npz_path  = os.path.join(dataset_dir, "compaction_data.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"compaction_data.npz not found at {npz_path}. "
                                "Run extract_dataset.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found at {meta_path}. "
                                "Run extract_dataset.py first.")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Verify GNS normalization stats are present
    if "normalization_stats" not in meta:
        raise ValueError(
            "metadata.json is missing 'normalization_stats'. "
            "Re-run extract_dataset.py with the updated version (Task A1)."
        )

    trajs = data["trajectories"]  # object array, each (T, N, 3)
    n_samples = len(trajs)
    print(f"Loaded {n_samples} trajectories. Sequence length = {trajs[0].shape[0]} steps, "
          f"Particle count range: {min(t.shape[1] for t in trajs)}–{max(t.shape[1] for t in trajs)}")

    # ------------------------------------------------------------------
    # 80 / 10 / 10 deterministic split (same seed as train.py)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    n_val   = int(0.1 * n_samples)
    n_test  = n_samples - n_train - n_val

    splits = {
        "train": indices[:n_train],
        "valid": indices[n_train : n_train + n_val],
        "test":  indices[n_train + n_val :],
    }
    print(f"Split sizes  ->  train: {n_train}  |  valid: {n_val}  |  test: {n_test}")

    # ------------------------------------------------------------------
    # Write per-split .npz files
    # ------------------------------------------------------------------
    gns_dir = os.path.join(dataset_dir, "gns")
    os.makedirs(gns_dir, exist_ok=True)
    for split_name, split_idx in splits.items():
        gns_data = []
        for k, i in enumerate(split_idx):
            traj = trajs[i].astype(np.float32)          # (T, n_p, 3)
            n_p  = traj.shape[1]
            p_type = np.ones(n_p, dtype=np.int32)       # uniform material type = 1
            gns_data.append((traj, p_type))

        out_path = os.path.join(gns_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, gns_data=np.array(gns_data, dtype=object))
        print(f"  Saved {split_name}.npz  ({len(split_idx)} trajectories)  ->  {out_path}")

    # ------------------------------------------------------------------
    # Write GNS-format metadata.json
    # ------------------------------------------------------------------
    # Box bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    box_w = 0.1    # 10 cm
    box_h = 0.25   # 25 cm initial height
    bounds = [
        [0.0, box_w],
        [0.0, box_w],
        [0.0, box_h],
    ]

    gns_meta = {
        "bounds":                      bounds,
        "sequence_length":             meta["sequence_length"],
        "default_connectivity_radius": meta["connectivity_radius"],
        "dim":                         3,
        "dt":                          1e-4,   # simulation timestep (seconds)
        "vel_mean":                    meta["normalization_stats"]["velocity"]["mean"],
        "vel_std":                     meta["normalization_stats"]["velocity"]["std"],
        "acc_mean":                    meta["normalization_stats"]["acceleration"]["mean"],
        "acc_std":                     meta["normalization_stats"]["acceleration"]["std"],
    }

    gns_meta_path = os.path.join(gns_dir, "metadata.json")
    with open(gns_meta_path, "w") as f:
        json.dump(gns_meta, f, indent=2)

    print(f"\n=== GNS dataset written to: {gns_dir} ===")
    print(f"    GNS metadata  : {gns_meta_path}")
    print(f"    Connectivity r: {gns_meta['default_connectivity_radius']} m")
    print(f"    Sequence len  : {gns_meta['sequence_length']} timestep snapshots")
    return gns_dir


if __name__ == "__main__":
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir  = os.path.join(base_dir, "dataset")
    convert_to_gns_format(dataset_dir)
