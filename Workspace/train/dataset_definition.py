from torch.utils.data import Dataset
from train.graph_creation import create_graph
from torch_geometric.utils import from_networkx
import torch

class ImageGraphDataset(Dataset):
    """
    Wraps an array of images and their corresponding NetworkX graphs.
    """
    def __init__(self, images, labels, segmenter='slic', **seg_kwargs):
        self.data    = images
        self.segmenter = segmenter
        self.seg_kwargs = seg_kwargs
        self.labels=labels
        for i, image in enumerate(self.data):
            G   = create_graph(image, method=self.segmenter, **self.seg_kwargs)
            graph = from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
            graph.y = self.labels[i]
            self.data[i] = graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
