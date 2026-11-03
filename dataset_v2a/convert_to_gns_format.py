"""
GNS Dataset Converter for Granular Compaction Data.
===================================================
Converts compaction_data.npz into the format expected by geoelements/gns:
  dataset/gns/
    ├── metadata.json   (GNS-format: bounds, shared connectivity_radius, train-only normalization stats)
    ├── train.npz       (same 38 training trajectories as PyG)
    ├── valid.npz       (same 5 validation trajectories as PyG)
    └── test.npz        (same 5 test trajectories as PyG)

Uses the exact split indices stored in dataset/metadata.json, ensuring identical
data partitioning between PyG GNN models and the GNS rollout simulator.
"""

import os
import json
import numpy as np

def convert_to_gns_format(dataset_dir: str) -> str:
    npz_path = os.path.join(dataset_dir, "compaction_data.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"compaction_data.npz not found at {npz_path}. Run extract_dataset.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found at {meta_path}. Run extract_dataset.py first.")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    trajs = data["trajectories"]  # object array, each (T, N, 3)
    n_samples = len(trajs)
    print(f"Loaded {n_samples} trajectories. Sequence length = {trajs[0].shape[0]} steps, "
          f"Particle count range: {min(t.shape[1] for t in trajs)}–{max(t.shape[1] for t in trajs)}")

    # Use the exact shared split indices from metadata.json
    splits = {
        "train": meta["split_indices"]["train"],
        "valid": meta["split_indices"]["valid"],
        "test":  meta["split_indices"]["test"],
    }
    print(f"Using shared split sizes -> train: {len(splits['train'])} | valid: {len(splits['valid'])} | test: {len(splits['test'])}")

    gns_dir = os.path.join(dataset_dir, "gns")
    os.makedirs(gns_dir, exist_ok=True)
    
    for split_name, split_idx in splits.items():
        gns_data = []
        for i in split_idx:
            traj = trajs[i].astype(np.float32)          # (T, n_p, 3)
            n_p  = traj.shape[1]
            p_type = np.ones(n_p, dtype=np.int32)       # uniform material type = 1
            gns_data.append((traj, p_type))

        out_path = os.path.join(gns_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, gns_data=np.array(gns_data, dtype=object))
        print(f"  Saved {split_name}.npz ({len(split_idx)} trajectories) -> {out_path}")

    # Write GNS metadata using the shared connectivity radius and box bounds
    box_w = 0.070    # 7.0 cm die box width
    box_h = 0.120    # 12.0 cm initial box height bounds
    bounds = [
        [0.0, box_w],
        [0.0, box_w],
        [0.0, box_h],
    ]

    gns_meta = {
        "bounds":                      bounds,
        "sequence_length":             meta["sequence_length"],
        "default_connectivity_radius": meta["connectivity_radius"], # 0.020 m (shared)
        "dim":                         3,
        "dt":                          5e-5,   # simulation timestep (seconds)
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
