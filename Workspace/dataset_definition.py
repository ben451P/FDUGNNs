from torch.utils.data import Dataset
from graph_creation import create_graph, create_graph_more_features
from torch_geometric.utils import from_networkx

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
