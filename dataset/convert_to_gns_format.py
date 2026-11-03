"""
GNS Dataset Converter for Granular Compaction Data.
===================================================
Converts compaction_data.npz into the format expected by geoelements/gns:
  dataset/gns/
    ├── metadata.json   (GNS-format: bounds, shared connectivity_radius, train-only normalization stats)
    ├── train.npz       (24 training trajectories)
    ├── valid.npz       (4 validation trajectories)
    └── test.npz        (4 test trajectories)

Enforces:
- Shared connectivity radius r_cutoff = 0.0105 m (1.05 * 2R = 10.5 mm)
- Normalization statistics computed strictly from training split
"""

import os
import json
import numpy as np

def convert_to_gns_format(dataset_dir: str) -> str:
    npz_path = os.path.join(dataset_dir, "compaction_data.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"compaction_data.npz not found at {npz_path}.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found at {meta_path}.")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    trajs = data["trajectories"]  # object array, each (17, N, 3)
    n_samples = len(trajs)
    seq_len = trajs[0].shape[0]
    print(f"Loaded {n_samples} trajectories. Sequence length = {seq_len} steps, "
          f"Particle count range: {min(t.shape[1] for t in trajs)}–{max(t.shape[1] for t in trajs)}")

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
            p_type = np.ones(n_p, dtype=np.int32)       # material type = 1
            gns_data.append((traj, p_type))

        out_path = os.path.join(gns_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, gns_data=np.array(gns_data, dtype=object))
        print(f"  Saved {split_name}.npz ({len(split_idx)} trajectories) -> {out_path}")

    # Compute normalization statistics strictly from training trajectories
    train_vels = []
    train_accs = []
    for i in splits["train"]:
        traj = trajs[i] # (T, N, 3)
        v = np.diff(traj, axis=0) # (T-1, N, 3)
        a = np.diff(v, axis=0)    # (T-2, N, 3)
        train_vels.append(v.reshape(-1, 3))
        train_accs.append(a.reshape(-1, 3))
        
    train_vels_flat = np.concatenate(train_vels, axis=0)
    train_accs_flat = np.concatenate(train_accs, axis=0)
    
    vel_mean = train_vels_flat.mean(axis=0).tolist()
    vel_std = (train_vels_flat.std(axis=0) + 1e-8).tolist()
    acc_mean = train_accs_flat.mean(axis=0).tolist()
    acc_std = (train_accs_flat.std(axis=0) + 1e-8).tolist()

    box_w = 0.070    # 7.0 cm die box width
    box_h = 0.120    # 12.0 cm initial box height bounds
    bounds = [
        [0.0, box_w],
        [0.0, box_w],
        [0.0, box_h],
    ]

    gns_meta = {
        "bounds":                      bounds,
        "sequence_length":             seq_len,
        "default_connectivity_radius": meta["connectivity_radius"], # 0.0105 m (shared)
        "dim":                         3,
        "dt":                          5e-5,
        "vel_mean":                    vel_mean,
        "vel_std":                     vel_std,
        "acc_mean":                    acc_mean,
        "acc_std":                     acc_std,
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
