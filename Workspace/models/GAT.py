import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear


class StaticGAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, edge_dim=None
    ):
        super(StaticGAT, self).__init__()
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        self.fc = Linear(hidden_channels, out_channels)  # Graph classification layer

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)  # Pool node features into graph features
        x = self.fc(x)  # Classify the entire graph
        return x

    def decode(self, z, edge_label_index, edge_attr=None):
        src, dst = edge_label_index
        edge_attr = (
            edge_attr
            if edge_attr is not None
            else torch.zeros(src.size(0), 1, device=z.device)
        )
        score = (z[src] * z[dst]).sum(dim=1) + edge_attr.squeeze()
        return score
