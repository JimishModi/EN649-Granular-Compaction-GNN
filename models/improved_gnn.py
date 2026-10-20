"""
Improved PyTorch Geometric GNN with Edge-Feature Embeddings and Residual Connections.
Enhances accuracy by integrating relative inter-particle distance vectors, residual skip connections,
and LayerNorm regularization tailored for moderate dataset sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class ResEdgeConvLayer(MessagePassing):
    """
    Residual Edge-Aware Message Passing Layer incorporating edge distance vectors.
    """
    def __init__(self, in_dim: int, edge_dim: int = 4):
        super(ResEdgeConvLayer, self).__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU()
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.layer_norm(x + self.update_mlp(torch.cat([x, out], dim=-1)))
        
    def message(self, x_i, x_j, edge_attr):
        e_emb = self.edge_mlp(edge_attr * 50.0)
        return self.msg_mlp(torch.cat([x_j + e_emb, x_i], dim=-1))


class ImprovedGNN(nn.Module):
    def __init__(self, in_dim=6, edge_dim=4, hidden_dim=48, global_dim=2, out_dim=2):
        super(ImprovedGNN, self).__init__()
        
        # Node Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Residual Message Passing Layers
        self.conv1 = ResEdgeConvLayer(hidden_dim, edge_dim=edge_dim)
        self.conv2 = ResEdgeConvLayer(hidden_dim, edge_dim=edge_dim)
        self.conv3 = ResEdgeConvLayer(hidden_dim, edge_dim=edge_dim)
        
        # Readout MLP
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        u = data.u
        
        # Node Encoding
        h = self.node_encoder(x)
        
        # Residual Message Passing
        h = self.conv1(h, edge_index, edge_attr)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.conv3(h, edge_index, edge_attr)
        
        # Global Mean Pooling
        graph_emb = global_mean_pool(h, batch)
        
        # Fuse global conditions (cohesion, displacement)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        out_features = torch.cat([graph_emb, u], dim=-1)
        
        # Final property prediction
        out = self.readout_mlp(out_features)
        return out


if __name__ == "__main__":
    model = ImprovedGNN()
    print("Improved GNN Architecture Refined:")
    print(model)
