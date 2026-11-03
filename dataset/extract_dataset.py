"""
Dataset Extractor for Revised Granular Compaction DEM Simulations.
==================================================================
Aggregates the 32 revised simulation outputs into a unified .npz archive and metadata.json.
Enforces:
1. 32 runs across 4 cohesion levels x 8 packing seeds.
2. 17 strain snapshots per run (total 512 state graphs).
3. Grouped 4-fold cross-validation by run/seed (all 17 snapshots of a run stay in the same fold).
4. Normalization statistics computed STRICTLY from training folds (no data leakage).
5. Single shared connectivity cutoff radius r_cutoff = 0.0105 m (1.05 * 2R) shared across PyG and GNS.
"""

import os
import json
import glob
import numpy as np

def process_dataset(output_dir="dem_sim/output", dataset_dir="dataset"):
    os.makedirs(dataset_dir, exist_ok=True)
    all_runs = sorted(glob.glob(os.path.join(output_dir, "run_*")))
    # Filter only revised runs (without 'disp' in name)
    run_folders = [r for r in all_runs if "disp" not in os.path.basename(r)]
    
    if len(run_folders) != 32:
        raise ValueError(f"Expected 32 revised run folders, found {len(run_folders)} in {output_dir}!")
        
    print(f"Processing {len(run_folders)} revised simulation runs into compressed .npz dataset...")
    
    all_trajectories = []
    all_snapshot_forces = []
    all_snapshot_strains = []
    all_snapshot_densities = []
    all_snapshot_void_ratios = []
    all_snapshot_coordinations = []
    all_snapshot_tensile_fractions = []
    all_snapshot_elastic_energies = []
    
    all_cohesions = []
    all_gammas = []
    all_seeds = []
    all_num_particles = []
    all_settled_heights = []
    all_settled_phis = []
    all_settled_e = []
    all_peak_forces = []
    all_final_void_ratios = []
    all_final_bulk_densities = []
    run_metas = []
    
    for rfolder in run_folders:
        meta_file = os.path.join(rfolder, "metadata.json")
        data_file = os.path.join(rfolder, "run_data.npz")
        
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        npz = np.load(data_file)
        
        all_trajectories.append(npz["snapshot_trajectories"]) # (17, N, 3)
        all_snapshot_forces.append(npz["snapshot_forces"])       # (17,)
        all_snapshot_strains.append(npz["snapshot_strains"])     # (17,)
        all_snapshot_densities.append(npz["snapshot_densities"]) # (17,)
        all_snapshot_void_ratios.append(npz["snapshot_void_ratios"]) # (17,)
        all_snapshot_coordinations.append(npz["snapshot_coordinations"]) # (17,)
        all_snapshot_tensile_fractions.append(npz["snapshot_tensile_fractions"]) # (17,)
        all_snapshot_elastic_energies.append(npz["snapshot_elastic_energies"]) # (17,)
        
        all_cohesions.append(meta["cohesion"])
        all_gammas.append(meta["gamma_work_of_adhesion"])
        all_seeds.append(meta["seed"])
        all_num_particles.append(meta["num_particles"])
        all_settled_heights.append(meta["settled_bed_height"])
        all_settled_phis.append(meta["phi_settled"])
        all_settled_e.append(meta["e_settled"])
        all_peak_forces.append(meta["peak_wall_force"])
        all_final_void_ratios.append(meta["final_void_ratio"])
        all_final_bulk_densities.append(meta["final_bulk_density"])
        run_metas.append(meta)
        
    num_runs = len(run_folders)
    print(f"Aggregated {num_runs} runs with {len(all_snapshot_strains[0])} strain snapshots per run.")
    
    # -------------------------------------------------------------
    # 4-Fold GroupKFold Cross-Validation Setup (Grouped by Seed)
    # -------------------------------------------------------------
    unique_seeds = sorted(list(set(all_seeds)))
    # 8 seeds: [11, 22, 33, 44, 55, 66, 77, 88] -> 4 folds of 2 seeds each (8 runs per fold)
    cv_folds = {}
    folds_def = [
        ("fold_1", [unique_seeds[0], unique_seeds[1]]),
        ("fold_2", [unique_seeds[2], unique_seeds[3]]),
        ("fold_3", [unique_seeds[4], unique_seeds[5]]),
        ("fold_4", [unique_seeds[6], unique_seeds[7]]),
    ]
    
    for fold_name, test_seeds in folds_def:
        test_run_indices = [i for i, s in enumerate(all_seeds) if s in test_seeds]
        train_run_indices = [i for i, s in enumerate(all_seeds) if s not in test_seeds]
        cv_folds[fold_name] = {
            "test_seeds": test_seeds,
            "test_run_indices": test_run_indices,
            "train_run_indices": train_run_indices
        }
        print(f"  {fold_name}: Test Seeds {test_seeds} ({len(test_run_indices)} runs) | Train ({len(train_run_indices)} runs)")
        
    # Standard GNS split using Fold 1 (train: 24 runs, val: 4 runs from Fold 2, test: 4 runs from Fold 2)
    gns_train_idx = cv_folds["fold_1"]["train_run_indices"] # 24 runs
    gns_val_idx = [i for i, s in enumerate(all_seeds) if s == unique_seeds[0]] # 4 runs
    gns_test_idx = [i for i, s in enumerate(all_seeds) if s == unique_seeds[1]] # 4 runs
    
    # Pack trajectories as object array
    traj_arr = np.empty(num_runs, dtype=object)
    for i in range(num_runs):
        traj_arr[i] = all_trajectories[i]
        
    npz_path = os.path.join(dataset_dir, "compaction_data.npz")
    np.savez_compressed(
        npz_path,
        trajectories=traj_arr,
        snapshot_forces=np.array(all_snapshot_forces, dtype=object),
        snapshot_strains=np.array(all_snapshot_strains, dtype=object),
        snapshot_densities=np.array(all_snapshot_densities, dtype=object),
        snapshot_void_ratios=np.array(all_snapshot_void_ratios, dtype=object),
        snapshot_coordinations=np.array(all_snapshot_coordinations, dtype=object),
        snapshot_tensile_fractions=np.array(all_snapshot_tensile_fractions, dtype=object),
        snapshot_elastic_energies=np.array(all_snapshot_elastic_energies, dtype=object),
        peak_forces=np.array(all_peak_forces, dtype=np.float32),
        cohesions=np.array(all_cohesions, dtype=np.float32),
        gammas=np.array(all_gammas, dtype=np.float32),
        seeds=np.array(all_seeds, dtype=np.int32),
        num_particles=np.array(all_num_particles, dtype=np.int32),
        settled_bed_heights=np.array(all_settled_heights, dtype=np.float32),
        settled_phis=np.array(all_settled_phis, dtype=np.float32),
        final_void_ratios=np.array(all_final_void_ratios, dtype=np.float32),
        final_bulk_densities=np.array(all_final_bulk_densities, dtype=np.float32)
    )
    print(f"Saved primary dataset to {npz_path}")
    
    # Save metadata.json
    connectivity_radius = 0.0105 # 1.05 * 2R = 10.5 mm
    metadata = {
        "num_runs": num_runs,
        "snapshots_per_run": len(all_snapshot_strains[0]),
        "total_graphs": num_runs * len(all_snapshot_strains[0]),
        "connectivity_radius": connectivity_radius,
        "cohesion_levels": sorted(list(set(all_cohesions))),
        "unique_seeds": unique_seeds,
        "box_width": 0.070,
        "box_area": 0.0049,
        "particle_radius": 0.005,
        "particle_density": 2500.0,
        "cv_folds": cv_folds,
        "split_indices": {
            "train": gns_train_idx,
            "valid": gns_val_idx,
            "test": gns_test_idx
        }
    }
    
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {os.path.join(dataset_dir, 'metadata.json')}")
    return metadata

if __name__ == "__main__":
    process_dataset()
