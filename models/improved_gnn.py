"""
Improved PyTorch Geometric GNN with Physics-Conditioned Message Passing & Dual Pooling.
========================================================================================
Matched Capacity: hidden_dim = 64, 3 message passing layers.
Physics-Conditioned Architecture:
- Encodes global condition vector u = [gamma, eps, N, H_settled] into 64-dim embedding h_u.
- Injects h_u DIRECTLY into message generation and node state updates at every layer,
  allowing message channels to modulate force transmission based on cohesion and strain.
- Dual-scale graph pooling: combines mean pooling (intensive properties: stress, density)
  with sum pooling (extensive properties: forces, energies).
- Predicts targets: [F(eps), rho_b(eps), e(eps)].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool

class ConditionedMsgLayer(MessagePassing):
    def __init__(self, hidden_dim=64, edge_dim=5):
        super(ConditionedMsgLayer, self).__init__(aggr='mean')
        # Message MLP conditioned on h_i, h_j, e_ij, and h_u
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Node update MLP conditioned on h_i, aggregated message, and h_u
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, h_u_expanded):
        # h_u_expanded is per-node or per-edge condition
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, h_u=h_u_expanded)
        return self.norm(x + self.node_mlp(torch.cat([x, out, h_u_expanded], dim=-1)))
        
    def message(self, x_i, x_j, edge_attr, h_u_j):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr, h_u_j], dim=-1))

class ImprovedGNN(nn.Module):
    def __init__(self, in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3):
        super(ImprovedGNN, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer1 = ConditionedMsgLayer(hidden_dim, edge_dim)
        self.layer2 = ConditionedMsgLayer(hidden_dim, edge_dim)
        self.layer3 = ConditionedMsgLayer(hidden_dim, edge_dim)
        
        # Dual-scale pooling readout (mean + sum/400 + h_u) -> (64 * 3 = 192 -> 64 -> 32 -> 3)
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        u = data.u
        if u.dim() == 1:
            u = u.unsqueeze(0)
            
        h_u = self.global_encoder(u) # (batch_size, hidden_dim)
        # Expand h_u to match node count
        h_u_nodes = h_u[batch] # (num_nodes, hidden_dim)
        
        h = self.node_encoder(x)
        h = self.layer1(h, edge_index, edge_attr, h_u_nodes)
        h = self.layer2(h, edge_index, edge_attr, h_u_nodes)
        h = self.layer3(h, edge_index, edge_attr, h_u_nodes)
        
        # Dual Pooling
        h_mean = global_mean_pool(h, batch)
        h_sum = global_add_pool(h, batch) / 400.0 # scale extensive sum
        
        combined = torch.cat([h_mean, h_sum, h_u], dim=-1)
        return self.readout_mlp(combined)

if __name__ == "__main__":
    from dataset import CompactionGraphDataset
    from torch_geometric.loader import DataLoader
    ds = CompactionGraphDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    model = ImprovedGNN()
    pred = model(batch)
    print("ImprovedGNN output shape:", pred.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ImprovedGNN total trainable parameters: {num_params}")
