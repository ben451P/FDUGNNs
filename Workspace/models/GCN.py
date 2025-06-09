import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, edge_dim=None):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is ignored in GCNConv
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
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
