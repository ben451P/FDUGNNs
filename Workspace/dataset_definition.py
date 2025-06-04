from torch.utils.data import Dataset
from graph_creation import create_graph, create_graph_more_features
from torch_geometric.utils import from_networkx
import torch

class ImageGraphDataset(Dataset):
    """
    Wraps an array of images and, on __getitem__, returns its corresponding NetworkX graph.
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
    
class ImageGraphDataset(Dataset):
    """
    Converts images to PyG graphs with extended node and edge features,
    and normalizes them across the dataset.
    """
    def __init__(self, images, labels, segmenter='slic', normalize=True, **seg_kwargs):
        self.segmenter = segmenter
        self.seg_kwargs = seg_kwargs
        self.labels = labels
        self.data = []

        # Use your new graph creation function
        for i, image in enumerate(images):
            G = create_graph(image, method=self.segmenter, **self.seg_kwargs)
            graph = from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
            graph.y = labels[i]
            # or more verbosely:

            self.data.append(graph)

        if normalize:
            self._normalize_attrs()

    def _normalize_attrs(self):
        all_node_feats = torch.cat([g.x for g in self.data], dim=0)
        all_edge_feats = torch.cat([g.edge_attr for g in self.data], dim=0)

        node_mean, node_std = all_node_feats.mean(dim=0), all_node_feats.std(dim=0)
        edge_mean, edge_std = all_edge_feats.mean(dim=0), all_edge_feats.std(dim=0)

        node_std[node_std == 0] = 1
        edge_std[edge_std == 0] = 1

        for g in self.data:
            g.x = (g.x - node_mean) / node_std
            g.edge_attr = (g.edge_attr - edge_mean) / edge_std

        self.node_mean, self.node_std = node_mean, node_std
        self.edge_mean, self.edge_std = edge_mean, edge_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# class ImageGraphDataset(Dataset):
#     """
#     Wraps an array of images and, on __getitem__, returns its corresponding NetworkX graph.
#     """
#     def __init__(self, images, labels, segmenter='slic', **seg_kwargs):
#         self.images    = images
#         self.segmenter = segmenter
#         self.seg_kwargs = seg_kwargs
#         self.labels=labels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = self.images[idx]
#         G   = create_graph(img, method=self.segmenter, **self.seg_kwargs)
#         G.y = self.labels[idx]
#         return G 
    
class ImageGraphDatasetRefined(Dataset):
    """
    Wraps an array of images and, on __getitem__, returns its corresponding NetworkX graph.
    """
    def __init__(self, images, labels, segmenter='slic', **seg_kwargs):
        self.images    = images
        self.segmenter = segmenter
        self.seg_kwargs = seg_kwargs
        self.labels=labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        G   = create_graph_more_features(img, method=self.segmenter, **self.seg_kwargs)
        return G, self.labels[idx]
