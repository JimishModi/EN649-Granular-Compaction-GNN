"""
Dataset Extractor for Granular Compaction DEM Simulations.
Aggregates simulation outputs into unified .npz archive and metadata.json.
"""

import os
import json
import glob
import numpy as np

def process_dataset(output_dir, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    run_folders = sorted(glob.glob(os.path.join(output_dir, "run_*")))
    
    if not run_folders:
        print("No simulation run folders found!")
        return None
        
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
        
    traj_arr = np.empty(len(all_trajectories), dtype=object)
    for i, t in enumerate(all_trajectories):
        traj_arr[i] = t

    vel_arr = np.empty(len(all_velocities), dtype=object)
    for i, v in enumerate(all_velocities):
        vel_arr[i] = v

    wf_arr = np.empty(len(all_wall_forces), dtype=object)
    for i, w in enumerate(all_wall_forces):
        wf_arr[i] = w

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
    
    # ------------------------------------------------------------------
    # Compute velocity and acceleration normalization stats for GNS
    # ------------------------------------------------------------------
    all_vels_flat = np.concatenate(
        [v.reshape(-1, 3) for v in all_velocities], axis=0
    ).astype(np.float64)

    all_accels_flat_list = []
    for v in all_velocities:
        accel = np.diff(v, axis=0)          # (T-1, N, 3)
        all_accels_flat_list.append(accel.reshape(-1, 3))
    all_accels_flat = np.concatenate(all_accels_flat_list, axis=0).astype(np.float64)

    vel_mean  = all_vels_flat.mean(axis=0).tolist()
    vel_std   = (all_vels_flat.std(axis=0) + 1e-8).tolist()
    acc_mean  = all_accels_flat.mean(axis=0).tolist()
    acc_std   = (all_accels_flat.std(axis=0) + 1e-8).tolist()

    seq_len = int(all_trajectories[0].shape[0])  # number of timestep snapshots

    overall_meta = {
        "num_samples": len(all_cohesions),
        "cohesion_levels": sorted(list(set(all_cohesions))),
        "displacement_levels": sorted(list(set(all_displacements))),
        "mean_void_ratio": float(np.mean(all_void_ratios)),
        "std_void_ratio": float(np.std(all_void_ratios)),
        "mean_bulk_density": float(np.mean(all_densities)),
        "std_bulk_density": float(np.std(all_densities)),
        "connectivity_radius": 0.015,    # 1.5x particle diameter
        "sequence_length": seq_len,
        "dim": 3,
        "normalization_stats": {
            "velocity": {
                "mean": vel_mean,
                "std":  vel_std
            },
            "acceleration": {
                "mean": acc_mean,
                "std":  acc_std
            }
        },
        "runs": run_metas
    }

    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(overall_meta, f, indent=2)

    print(f"=== Dataset extracted successfully! Saved to {npz_path} ({os.path.getsize(npz_path)/1024:.1f} KB) ===")
    print(f"    Sequence length : {seq_len} snapshots")
    print(f"    Velocity  mean  : {[f'{v:.4e}' for v in vel_mean]}")
    print(f"    Velocity  std   : {[f'{v:.4e}' for v in vel_std]}")
    print(f"    Accel     mean  : {[f'{v:.4e}' for v in acc_mean]}")
    print(f"    Accel     std   : {[f'{v:.4e}' for v in acc_std]}")
    return npz_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "dem_sim", "output")
    dataset_dir = os.path.join(base_dir, "dataset")
    process_dataset(output_dir, dataset_dir)
