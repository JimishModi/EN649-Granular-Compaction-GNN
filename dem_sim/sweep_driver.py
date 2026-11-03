"""
Multi-core Sweep Driver for Granular DEM Compaction Simulations.
================================================================
Executes the revised 32-run parameter matrix:
- 4 Cohesion levels:   {0.0, 50.0, 100.0, 150.0} kPa
- 8 Packing seeds:     {11, 22, 33, 44, 55, 66, 77, 88}
- Compaction to 13% axial strain at constant piston velocity v = 0.05 m/s
- Logging 16 strain snapshots per run (total 512 state graphs across sweep)
Total = 4 x 8 = 32 runs.

Particle count is a deterministic function of seed:
N(seed_idx) = 380 + seed_idx * 10 in [380, 450].
"""

import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from python_dem import ParticleBedDEM

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]

def get_deterministic_particle_count(seed):
    """
    Deterministic function mapping seed to particle count.
    Avoids unseeded random variations while ensuring realistic physical variance across beds.
    """
    if seed in SEEDS:
        idx = SEEDS.index(seed)
        return 380 + idx * 10 # 380, 390, 400, 410, 420, 430, 440, 450
    return 380 + (int(seed) % 8) * 10

def run_single_simulation(run_info):
    """
    Executes a single simulation instance with given parameters and saves outputs.
    """
    run_id, cohesion, max_strain, seed, num_particles, output_dir = run_info
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    sim = ParticleBedDEM(
        num_particles=num_particles,
        cohesion=cohesion,
        max_strain=max_strain,
        seed=seed,
        piston_velocity=0.05
    )
    res = sim.run_simulation(settle_steps=4000, num_snapshots=16)
    elapsed = time.time() - start_time
    
    # Save trajectory and velocity arrays
    np.savez_compressed(
        os.path.join(output_dir, "run_data.npz"),
        settled_positions=res["settled_positions"],
        snapshot_strains=res["snapshot_strains"],
        snapshot_forces=res["snapshot_forces"],
        snapshot_stresses=res["snapshot_stresses"],
        snapshot_trajectories=res["snapshot_trajectories"],
        snapshot_coordinations=res["snapshot_coordinations"],
        snapshot_tensile_fractions=res["snapshot_tensile_fractions"],
        snapshot_elastic_energies=res["snapshot_elastic_energies"],
        snapshot_inertial_numbers=res["snapshot_inertial_numbers"],
        snapshot_void_ratios=res["snapshot_void_ratios"],
        snapshot_densities=res["snapshot_densities"],
        ke_history=res["ke_history"]
    )
    
    # Save run metadata
    meta = {
        "run_id": run_id,
        "cohesion": float(cohesion),
        "max_strain": float(max_strain),
        "seed": int(seed),
        "num_particles": int(num_particles),
        "settled_bed_height": float(res["settled_bed_height"]),
        "phi_settled": float(res["phi_settled"]),
        "e_settled": float(res["e_settled"]),
        "settled_overlap_ratio": float(res["settled_overlap_ratio"]),
        "settled_coordination": float(res["settled_coordination"]),
        "final_piston_height": float(res["final_piston_height"]),
        "final_void_ratio": float(res["final_void_ratio"]),
        "final_bulk_density": float(res["final_bulk_density"]),
        "peak_wall_force": float(res["peak_wall_force"]),
        "peak_axial_stress": float(res["peak_axial_stress"]),
        "final_coordination": float(res["final_coordination"]),
        "final_tensile_fraction": float(res["final_tensile_fraction"]),
        "final_elastic_energy": float(res["final_elastic_energy"]),
        "mean_inertial_number": float(res["mean_inertial_number"]),
        "gamma_work_of_adhesion": float(res["gamma_work_of_adhesion"]),
        "f_pull_off": float(res["f_pull_off"]),
        "bond_number": float(res["bond_number"]),
        "settle_wall_time_sec": float(res["settle_wall_time_sec"]),
        "compact_wall_time_sec": float(res["compact_wall_time_sec"]),
        "elapsed_seconds": float(elapsed),
        "box_bounds": res["box_bounds"]
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    return {
        "status": "SUCCESS",
        "engine": "PythonDEM",
        "run_id": run_id,
        "elapsed": elapsed,
        "output_dir": output_dir,
        "meta": meta
    }

def execute_sweep(num_workers=4):
    cohesion_levels = [0.0, 50.0, 100.0, 150.0]        # 4 cohesion levels (kPa)
    max_strain = 0.13                                  # 13% axial strain
    seeds = SEEDS                                      # 8 seeds
    
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(base_output_dir, exist_ok=True)
    
    tasks = []
    run_counter = 0
    for coh in cohesion_levels:
        for seed in seeds:
            run_counter += 1
            run_id = f"run_{run_counter:03d}_coh{int(coh)}_seed{seed}"
            out_dir = os.path.join(base_output_dir, run_id)
            num_particles = get_deterministic_particle_count(seed)
            tasks.append((run_id, coh, max_strain, seed, num_particles, out_dir))
            
    print(f"--- Launching Revised DEM Simulation Sweep ({len(tasks)} runs across {num_workers} parallel workers) ---")
    start_total = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_simulation, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            meta = res["meta"]
            print(f"[{idx:02d}/{len(tasks):02d}] {res['run_id']} finished in {res['elapsed']:.2f}s | "
                  f"N={meta['num_particles']} | H_set={meta['settled_bed_height']*100:.1f}cm | "
                  f"phi={meta['phi_settled']:.3f} | e={meta['final_void_ratio']:.3f} | "
                  f"Peak F={meta['peak_wall_force']:.1f}N | I={meta['mean_inertial_number']:.2e}")
                  
    total_elapsed = time.time() - start_total
    print(f"\nAll {len(results)} simulations completed successfully in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)!")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    args = parser.parse_args()
    execute_sweep(num_workers=args.workers)
