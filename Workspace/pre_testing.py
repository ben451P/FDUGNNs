import os
import torch
import torch.optim as optim
import pandas as pd
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
import torch_directml

print(torch_directml.device())


img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
img_path = os.path.join(img_dir, "ISIC_0015719.jpg")
img_path = [io.imread(img_path)]
labels = [0]

def graph_collate(batch):
    data_list = []
    for G, y in batch:
        data = from_networkx(G)
        # pack all node features into x
        feats = torch.tensor(
            [[G.nodes[n]['mean'], G.nodes[n]['median'], G.nodes[n]['centroid_x'], G.nodes[n]['centroid_y']] for n in G.nodes],
            dtype=torch.float
        )
        data.x = feats
        # pack edge attributes if present
        if G.number_of_edges() > 0 and 'diff_mean' in list(G.edges(data=True))[0][2]:
            edge_attrs = torch.tensor(
                [[attr.get('diff_mean',0), attr.get('diff_median',0)] for _,_,attr in G.edges(data=True)],
                dtype=torch.float
            )
            data.edge_attr = edge_attrs
        data.y = torch.tensor([y], dtype=torch.long)
        data_list.append(data)
    return Batch.from_data_list(data_list)

dataset = ImageGraphDataset(img_path, labels, segmenter='slic', n_segments=100)
print(dataset[0])
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in train_loader:
    print(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
# added an edge dim (2)
model = EDGAT(4, 32, 2, 1, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x = train(model, optimizer, train_loader)
print(x)
stats = test(model, train_loader)
print(stats)