"""
Baseline PyTorch Geometric GNN for Granular Compaction Property Regression.
Predicts final bulk density and void ratio directly from initial particle graph state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BaselineGNN(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, global_dim=2, out_dim=2):
        super(BaselineGNN, self).__init__()
        # Node Feature Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message Passing Layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Readout MLP (combines pooled node representations + global inputs [cohesion, displacement])
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        u = data.u # (batch_size, global_dim)
        
        # Encode nodes
        h = self.node_encoder(x)
        
        # Message passing layers
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        
        # Readout (Global Mean Pooling)
        graph_emb = global_mean_pool(h, batch)
        
        # Fuse global conditions (cohesion, displacement)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        out_features = torch.cat([graph_emb, u], dim=-1)
        
        # Final property prediction
        out = self.readout_mlp(out_features)
        return out

if __name__ == "__main__":
    from dataset import CompactionGraphDataset
    from torch_geometric.loader import DataLoader
    
    model = BaselineGNN()
    print("Baseline GNN Architecture:")
    print(model)
