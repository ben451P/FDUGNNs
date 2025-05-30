from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from skimage import io
from torch.utils.data import Dataset
from new_file import create_graph_more_features
import os
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
from mainloop_functions  import train, test
from models.DGAT import DynamicGAT as DGAT
from models.EDGAT import DynamicEdgeGAT as EDGAT
from models.GAT import StaticGAT as GAT
from dataset_definition import ImageGraphDataset
from torch.utils.data import random_split
from skimage import io

# class ImageGraphDataset(Dataset):
#     def __init__(self, images, labels, segmenter='slic', normalize=True, **seg_kwargs):
#         self.data = []
#         self.segmenter = segmenter
#         self.seg_kwargs = seg_kwargs
#         self.labels = labels
#         self.normalize = normalize
        
#         # First pass to collect all node features for statistics
#         if self.normalize:
#             all_features = []
#             for image in images:
#                 G = create_graph_more_features(image, method=self.segmenter, **self.seg_kwargs)
#                 all_features.append(np.array([data['x'] for _, data in G.nodes(data=True)]))
#             all_features = np.concatenate(all_features)
#             self.mean = np.mean(all_features, axis=0)
#             self.std = np.std(all_features, axis=0)
        
#         # Second pass to create normalized graphs
#         for i, image in enumerate(images):
#             G = create_graph_more_features(image, method=self.segmenter, **self.seg_kwargs)
            
#             if self.normalize:
#                 # Normalize node features
#                 for _, data in G.nodes(data=True):
#                     data['x'] = (np.array(data['x']) - self.mean) / (self.std + 1e-8)
            
#             graph = from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
#             graph.y = self.labels[i]
#             self.data.append(graph)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

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
            G   = create_graph_more_features(image, method=self.segmenter, **self.seg_kwargs)
            graph = from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
            graph.y = self.labels[i]
            self.data[i] = graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
img_path = os.path.join(img_dir, "ISIC_0015719.jpg")
img_path = [io.imread(img_path)]
labels = [0]

dataset = ImageGraphDataset(img_path, labels, segmenter='slic', n_segments=100)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in train_loader:
    print(batch)
    print(batch[0].x, batch[0].edge_attr)

model = EDGAT(8, 32, 2, 1, 4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x = train(model, optimizer, train_loader)
print(x)
stats = test(model, train_loader)
print(stats)