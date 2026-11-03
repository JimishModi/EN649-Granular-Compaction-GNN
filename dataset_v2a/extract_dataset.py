"""
Dataset Extractor for Granular Compaction DEM Simulations.
==========================================================
Aggregates 48 simulation outputs into a unified .npz archive and metadata.json.
Enforces:
1. Deterministic 80/10/10 train/val/test split saved in metadata.json so all models
   (PyG BaselineGNN, ImprovedGNN, and GNS) share the exact same split.
2. Normalization statistics (velocity, acceleration, density, void ratio) computed
   STRICTLY from the training split runs only (no data leakage).
3. A single documented connectivity radius (0.020 m = 2x particle diameter)
   shared across PyG and GNS.
"""

import os
import json
import glob
import numpy as np

def process_dataset(output_dir, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    run_folders = sorted(glob.glob(os.path.join(output_dir, "run_*")))
    
    if not run_folders:
        raise FileNotFoundError(f"No simulation run folders found in {output_dir}!")
        
    print(f"Processing {len(run_folders)} simulation runs into compressed .npz dataset...")
    
    all_trajectories = []
    all_velocities = []
    all_wall_forces = []
    all_cohesions = []
    all_displacements = []
    all_void_ratios = []
    all_densities = []
    all_particle_types = []
    run_metas = []
    
    for rfolder in run_folders:
        meta_file = os.path.join(rfolder, "metadata.json")
        data_file = os.path.join(rfolder, "run_data.npz")
        
        if not (os.path.exists(meta_file) and os.path.exists(data_file)):
            continue
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        npz = np.load(data_file)
        traj = npz["trajectory"]   # (T, N, 3)
        vel = npz["velocities"]    # (T, N, 3)
        wforces = npz["wall_forces"]
        
        all_trajectories.append(traj)
        all_velocities.append(vel)
        all_wall_forces.append(wforces)
        all_cohesions.append(meta["cohesion"])
        all_displacements.append(meta["displacement"])
        all_void_ratios.append(meta["final_void_ratio"])
        all_densities.append(meta["final_bulk_density"])
        all_particle_types.append(np.ones(meta["num_particles"], dtype=np.int64))
        run_metas.append(meta)
        
    num_samples = len(all_trajectories)
    print(f"Aggregated {num_samples} valid simulation runs.")
    
    # -------------------------------------------------------------
    # 80 / 10 / 10 Deterministic Split
    # -------------------------------------------------------------
    rng = np.random.default_rng(42)
    indices = rng.permutation(num_samples).tolist()
    n_train = int(np.round(0.80 * num_samples)) # 38
    n_val = int(np.round(0.10 * num_samples))   # 5
    n_test = num_samples - n_train - n_val      # 5
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"Split distribution: Train={len(train_indices)} (80%), Val={len(val_indices)} (10%), Test={len(test_indices)} (10%)")
    print(f"  Train indices: {train_indices}")
    print(f"  Val indices:   {val_indices}")
    print(f"  Test indices:  {test_indices}")
    
    # Pack trajectory, velocity, wall forces as object arrays (variable particle count per seed)
    traj_arr = np.empty(num_samples, dtype=object)
    vel_arr = np.empty(num_samples, dtype=object)
    wf_arr = np.empty(num_samples, dtype=object)
    
    for i in range(num_samples):
        traj_arr[i] = all_trajectories[i]
        vel_arr[i] = all_velocities[i]
        wf_arr[i] = all_wall_forces[i]
        
    npz_path = os.path.join(dataset_dir, "compaction_data.npz")
    np.savez_compressed(
        npz_path,
        trajectories=traj_arr,
        velocities=vel_arr,
        wall_forces=wf_arr,
        cohesion=np.array(all_cohesions, dtype=np.float32),
        displacement=np.array(all_displacements, dtype=np.float32),
        final_void_ratio=np.array(all_void_ratios, dtype=np.float32),
        final_bulk_density=np.array(all_densities, dtype=np.float32)
    )
    
    # -------------------------------------------------------------
    # Normalization Statistics Computed from TRAINING RUNS ONLY
    # -------------------------------------------------------------
    train_vels_flat = np.concatenate(
        [all_velocities[idx].reshape(-1, 3) for idx in train_indices], axis=0
    ).astype(np.float64)
    
    train_accels_flat_list = []
    for idx in train_indices:
        v = all_velocities[idx]
        accel = np.diff(v, axis=0) # (T-1, N, 3)
        train_accels_flat_list.append(accel.reshape(-1, 3))
    train_accels_flat = np.concatenate(train_accels_flat_list, axis=0).astype(np.float64)
    
    vel_mean = train_vels_flat.mean(axis=0).tolist()
    vel_std = (train_vels_flat.std(axis=0) + 1e-8).tolist()
    acc_mean = train_accels_flat.mean(axis=0).tolist()
    acc_std = (train_accels_flat.std(axis=0) + 1e-8).tolist()
    
    train_void_ratios = [all_void_ratios[idx] for idx in train_indices]
    train_densities = [all_densities[idx] for idx in train_indices]
    
    seq_len = int(all_trajectories[0].shape[0])
    connectivity_radius = 0.020 # 2x particle diameter = 4x radius (shared with PyG and GNS)
    
    overall_meta = {
        "num_samples": num_samples,
        "cohesion_levels": sorted(list(set(all_cohesions))),
        "displacement_levels": sorted(list(set(all_displacements))),
        "mean_void_ratio": float(np.mean(all_void_ratios)),
        "std_void_ratio": float(np.std(all_void_ratios)),
        "mean_bulk_density": float(np.mean(all_densities)),
        "std_bulk_density": float(np.std(all_densities)),
        "connectivity_radius": connectivity_radius,
        "sequence_length": seq_len,
        "dim": 3,
        "split_indices": {
            "train": train_indices,
            "valid": val_indices,
            "test": test_indices
        },
        "train_target_stats": {
            "density_mean": float(np.mean(train_densities)),
            "density_std": float(np.std(train_densities)),
            "void_ratio_mean": float(np.mean(train_void_ratios)),
            "void_ratio_std": float(np.std(train_void_ratios))
        },
        "normalization_stats": {
            "velocity": {
                "mean": vel_mean,
                "std": vel_std
            },
            "acceleration": {
                "mean": acc_mean,
                "std": acc_std
            }
        },
        "runs": run_metas
    }
    
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(overall_meta, f, indent=2)
        
    print(f"=== Dataset extracted successfully! Saved to {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB) ===")
    print(f"    Connectivity radius: {connectivity_radius} m (shared across PyG and GNS)")
    print(f"    Train Target Stats  : Density = {overall_meta['train_target_stats']['density_mean']:.2f} +/- {overall_meta['train_target_stats']['density_std']:.2f} kg/m^3")
    print(f"                          Void Ratio = {overall_meta['train_target_stats']['void_ratio_mean']:.4f} +/- {overall_meta['train_target_stats']['void_ratio_std']:.4f}")
    return npz_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "dem_sim", "output")
    dataset_dir = os.path.join(base_dir, "dataset")
    process_dataset(output_dir, dataset_dir)
