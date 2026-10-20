"""
PyTorch Geometric Dataset Class for Granular Particle Bed Compaction.
"""

import os
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from scipy.spatial import cKDTree

class CompactionGraphDataset(Dataset):
    def __init__(self, npz_path, radius=0.020):  # 0.020 = 2x particle diameter
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.trajectories = data["trajectories"] # (num_samples, T, N, 3)
        self.velocities = data["velocities"]     # (num_samples, T, N, 3)
        self.cohesions = data["cohesion"]
        self.displacements = data["displacement"]
        self.densities = data["final_bulk_density"]
        self.void_ratios = data["final_void_ratio"]
        self.radius = radius
        
    def len(self):
        return len(self.cohesions)
        
    def get(self, idx):
        # Initial particle positions and velocities
        init_pos = self.trajectories[idx][0] # (N, 3)
        init_vel = self.velocities[idx][0]   # (N, 3)
        
        # Build radius graph edges using cKDTree
        tree = cKDTree(init_pos)
        pairs = tree.query_pairs(r=self.radius)
        
        if len(pairs) == 0:
            # Fallback to k-NN if radius yields no edges
            _, idxs = tree.query(init_pos, k=4)
            edge_list = []
            for i in range(len(init_pos)):
                for j in idxs[i][1:]:
                    edge_list.append([i, j])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_list = []
            for i, j in pairs:
                edge_list.append([i, j])
                edge_list.append([j, i]) # Bidirectional
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
        # Node features: [x, y, z, vx, vy, vz]
        node_features = np.hstack([init_pos, init_vel])
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge features: relative displacement vector [dx, dy, dz] and scalar distance r
        row, col = edge_index[0], edge_index[1]
        rel_pos = torch.tensor(init_pos[col] - init_pos[row], dtype=torch.float)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, dist], dim=-1)
        
        # Global features: [cohesion, displacement]
        u = torch.tensor([self.cohesions[idx], self.displacements[idx]], dtype=torch.float).unsqueeze(0)
        
        # Targets: [final_bulk_density, final_void_ratio]
        y = torch.tensor([self.densities[idx], self.void_ratios[idx]], dtype=torch.float).unsqueeze(0)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "compaction_data.npz")
    if os.path.exists(dataset_path):
        ds = CompactionGraphDataset(dataset_path)
        sample = ds[0]
        print(f"Dataset sample 0 loaded successfully: Nodes = {sample.x.shape}, Edges = {sample.edge_index.shape}, Targets = {sample.y}")
