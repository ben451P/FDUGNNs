from torch.utils.data import Dataset
from graph_creation import create_graph, create_graph_more_features

class ImageGraphDataset(Dataset):
    """
    Wraps an array of images and, on __getitem__, returns its corresponding NetworkX graph.
    """
    def __init__(self, images, segmenter='slic', **seg_kwargs):
        self.images    = images
        self.segmenter = segmenter
        self.seg_kwargs = seg_kwargs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        G   = create_graph(img, method=self.segmenter, **self.seg_kwargs)
        return G
