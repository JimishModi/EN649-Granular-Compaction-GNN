"""
Multi-core Sweep Driver for DEM Compaction Simulations.
Sweeps cohesion levels, piston displacements, and packing seeds.
Supports both LAMMPS binary (in WSL) and high-performance Python DEM fallback.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to import python_dem if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from python_dem import ParticleBedDEM

def run_single_simulation(run_info):
    """
    Executes a single simulation instance with given parameters.
    """
    run_id, cohesion, displacement, seed, num_particles, output_dir, use_lammps = run_info
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    if use_lammps:
        # Construct LAMMPS command inside WSL
        cmd = f"lammps -var COHESION {cohesion} -var DISPLACEMENT {displacement} -var SEED {seed} -var NUM_PARTICLES {num_particles} -var OUTPUT_DIR {output_dir} -in dem_sim/in.compaction.template"
        try:
            res = subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True, text=True, timeout=120)
            elapsed = time.time() - start_time
            if res.returncode == 0:
                return {"status": "SUCCESS", "engine": "LAMMPS", "run_id": run_id, "elapsed": elapsed, "output_dir": output_dir}
        except Exception:
            pass
            
    # Python DEM Fallback
    sim = ParticleBedDEM(num_particles=num_particles, cohesion=cohesion, displacement=displacement, seed=seed)
    res = sim.run_simulation(total_steps=150, dt=1e-4, dump_interval=5)
    elapsed = time.time() - start_time
    
    # Save results to output_dir in numpy/json format
    np.savez_compressed(
        os.path.join(output_dir, "run_data.npz"),
        trajectory=res["trajectory"],
        velocities=res["velocities"],
        wall_forces=res["wall_forces"]
    )
    
    meta = {
        "run_id": run_id,
        "cohesion": cohesion,
        "displacement": displacement,
        "seed": seed,
        "num_particles": num_particles,
        "final_void_ratio": res["final_void_ratio"],
        "final_bulk_density": res["final_bulk_density"],
        "elapsed_seconds": elapsed,
        "box_bounds": res["box_bounds"]
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    return {"status": "SUCCESS", "engine": "PythonDEM", "run_id": run_id, "elapsed": elapsed, "output_dir": output_dir, "meta": meta}


def execute_sweep(num_workers=4, use_lammps=False):
    cohesion_levels = [0.0, 50.0, 150.0]        # 3 cohesion levels ("materials")
    displacement_levels = [0.15, 0.25, 0.35, 0.45] # 4 displacement ratios
    seeds = [42, 101, 202, 303]                # 4 seeds
    
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(base_output_dir, exist_ok=True)
    
    tasks = []
    run_counter = 0
    for coh in cohesion_levels:
        for disp in displacement_levels:
            for seed in seeds:
                run_counter += 1
                run_id = f"run_{run_counter:03d}_coh{int(coh)}_disp{int(disp*100)}_seed{seed}"
                out_dir = os.path.join(base_output_dir, run_id)
                num_particles = np.random.randint(350, 550)
                tasks.append((run_id, coh, disp, seed, num_particles, out_dir, use_lammps))
                
    print(f"--- Launching DEM Simulation Sweep ({len(tasks)} runs across {num_workers} parallel workers) ---")
    start_total = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_simulation, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            print(f"[{idx}/{len(tasks)}] Completed Run {res['run_id']} ({res['engine']}) in {res['elapsed']:.2f}s")
            
    total_time = time.time() - start_total
    print(f"\n=== DEM Sweep Complete! Total Time: {total_time:.2f} seconds ===")
    return results

if __name__ == "__main__":
    execute_sweep(num_workers=4, use_lammps=False)
