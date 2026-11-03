"""
PyTorch Geometric Dataset for Revised Granular Compaction DEM Simulations.
===========================================================================
Precomputes all 512 graph snapshots (32 runs x 17 strain increments) in memory for O(1) access.
Features:
- Node features (5-dim): [x, y, z, d_bot, d_lat] (spatial position and boundary clearances)
- Edge features (5-dim): [dx/r_c, dy/r_c, dz/r_c, r/r_c, delta_n/R] (contact overlap mechanics)
- Global vector u (4-dim): [gamma/4.5, eps/0.13, (N-380)/70, (H_set-0.08)/0.02]
- Targets y (3-dim): [F(eps), rho_b(eps), e(eps)]
- Shared physical connectivity cutoff radius r_cutoff = 0.0105 m (1.05 * 2R)
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial import cKDTree

def build_all_compaction_graphs(npz_path: str = "dataset/compaction_data.npz", radius: float = 0.0105):
    data = np.load(npz_path, allow_pickle=True)
    trajectories = data["trajectories"] # (32,) object array of (17, N, 3)
    snapshot_forces = data["snapshot_forces"] # (32,) object array of (17,)
    snapshot_strains = data["snapshot_strains"] # (32,) object array of (17,)
    snapshot_densities = data["snapshot_densities"] # (32,) object array of (17,)
    snapshot_void_ratios = data["snapshot_void_ratios"] # (32,) object array of (17,)
    
    peak_forces = data["peak_forces"] # (32,)
    cohesions = data["cohesions"]
    gammas = data["gammas"]
    seeds = data["seeds"]
    num_particles = data["num_particles"]
    settled_heights = data["settled_bed_heights"]
    
    radius_cutoff = float(radius) # 0.0105 m
    particle_radius = 0.005 # 5 mm
    box_width = 0.070 # 7 cm
    
    graph_list = []
    num_runs = len(cohesions)
    
    for r_idx in range(num_runs):
        n_p = int(num_particles[r_idx])
        num_snaps = len(snapshot_strains[r_idx])
        gamma_val = float(gammas[r_idx])
        n_val = float(num_particles[r_idx])
        h_set_val = float(settled_heights[r_idx])
        peak_f_val = float(peak_forces[r_idx])
        
        for s_idx in range(num_snaps):
            pos = trajectories[r_idx][s_idx].astype(np.float32) # (N, 3)
            eps_val = float(snapshot_strains[r_idx][s_idx])
            f_val = float(snapshot_forces[r_idx][s_idx])
            rho_val = float(snapshot_densities[r_idx][s_idx])
            e_val = float(snapshot_void_ratios[r_idx][s_idx])
            
            # KDTree contact search
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=radius_cutoff, output_type='ndarray')
            
            if len(pairs) == 0:
                _, idxs = tree.query(pos, k=min(4, n_p))
                edge_list = []
                for i in range(n_p):
                    for j in idxs[i][1:]:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                i = pairs[:, 0]
                j = pairs[:, 1]
                row = np.concatenate([i, j])
                col = np.concatenate([j, i])
                edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
                
            # Node features (5-dim)
            d_bot = pos[:, 2] - particle_radius
            d_lat = np.minimum.reduce([
                pos[:, 0] - particle_radius,
                box_width - particle_radius - pos[:, 0],
                pos[:, 1] - particle_radius,
                box_width - particle_radius - pos[:, 1]
            ])
            node_feat = np.column_stack([pos, d_bot, d_lat])
            x = torch.tensor(node_feat, dtype=torch.float)
            
            # Edge features (5-dim)
            row_t, col_t = edge_index[0], edge_index[1]
            rel_pos = torch.tensor(pos[col_t] - pos[row_t], dtype=torch.float)
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)
            overlap = torch.clamp(2.0 * particle_radius - dist, min=0.0) / particle_radius
            edge_attr = torch.cat([rel_pos / radius_cutoff, dist / radius_cutoff, overlap], dim=-1)
            
            # Global context vector u (4-dim)
            u_vec = [
                gamma_val / 4.5,
                eps_val / 0.13,
                (n_val - 380.0) / 70.0,
                (h_set_val - 0.08) / 0.02
            ]
            u = torch.tensor(u_vec, dtype=torch.float).unsqueeze(0)
            
            # Targets: [F(eps), rho_b(eps), e(eps)]
            y = torch.tensor([f_val, rho_val, e_val], dtype=torch.float).unsqueeze(0)
            
            data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)
            data_obj.run_idx = r_idx
            data_obj.snapshot_idx = s_idx
            data_obj.seed = int(seeds[r_idx])
            data_obj.cohesion = float(cohesions[r_idx])
            data_obj.gamma = gamma_val
            data_obj.strain = eps_val
            data_obj.num_particles = int(n_val)
            data_obj.settled_height = h_set_val
            data_obj.peak_force = peak_f_val
            
            graph_list.append(data_obj)
            
    return graph_list

class CompactionGraphDataset(Dataset):
    def __init__(self, npz_path: str = "dataset/compaction_data.npz", radius: float = 0.0105):
        super().__init__()
        self.data_list = build_all_compaction_graphs(npz_path, radius)
        
    def len(self):
        return len(self.data_list)
        
    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    import time
    t0 = time.time()
    ds = CompactionGraphDataset()
    print(f"Precomputed {len(ds)} graph snapshots in {time.time()-t0:.2f} seconds.")
    d0 = ds[0]
    print(f"Graph 0: Nodes={d0.num_nodes}, Edges={d0.num_edges}, x={d0.x.shape}, e={d0.edge_attr.shape}, u={d0.u.shape}, y={d0.y.shape}")
