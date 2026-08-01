"""
GIN model for graph classification (predicting a label for a whole graph,
not individual nodes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # combine all node embeddings in each graph into one vector per graph
        graph_embedding = global_add_pool(x, batch)
        return self.classifier(graph_embedding)
