"""
Baseline PyTorch Geometric GNN for Granular Compaction Regression.
===================================================================
Matched Capacity: hidden_dim = 64, 3 message passing layers.
Decoupled Architecture:
- Message passing resolves inter-particle topology and contact geometry.
- Global conditioning u = [gamma, eps, N, H_settled] is excluded from message passing
  and fused solely at the global readout MLP.
- Predicts targets: [F(eps), rho_b(eps), e(eps)].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class BaselineMsgLayer(MessagePassing):
    def __init__(self, hidden_dim=64, edge_dim=5):
        super(BaselineMsgLayer, self).__init__(aggr='mean')
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(x + self.node_mlp(torch.cat([x, out], dim=-1)))
        
    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

class BaselineGNN(nn.Module):
    def __init__(self, in_dim=5, edge_dim=5, hidden_dim=64, global_dim=4, out_dim=3):
        super(BaselineGNN, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer1 = BaselineMsgLayer(hidden_dim, edge_dim)
        self.layer2 = BaselineMsgLayer(hidden_dim, edge_dim)
        self.layer3 = BaselineMsgLayer(hidden_dim, edge_dim)
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, 64),
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
            
        h = self.node_encoder(x)
        h = self.layer1(h, edge_index, edge_attr)
        h = self.layer2(h, edge_index, edge_attr)
        h = self.layer3(h, edge_index, edge_attr)
        
        # Mean pooling
        graph_emb = global_mean_pool(h, batch)
        
        # Late fusion of global condition u
        out_features = torch.cat([graph_emb, u], dim=-1)
        return self.readout_mlp(out_features)

if __name__ == "__main__":
    from dataset import CompactionGraphDataset
    from torch_geometric.loader import DataLoader
    ds = CompactionGraphDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    model = BaselineGNN()
    pred = model(batch)
    print("BaselineGNN output shape:", pred.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BaselineGNN total trainable parameters: {num_params}")
