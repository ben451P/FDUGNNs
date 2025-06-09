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
from models.GCN import GCN
from dataset_definition import ImageGraphDataset
from torch.utils.data import random_split
from skimage import io
import torch_directml
from preprocessing import stratified_split

print(torch_directml.device())


img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
img_path = os.path.join(img_dir, "ISIC_0015719.jpg")
img_path = [io.imread(img_path)] * 20
labels = [0] * 20

segmentation_algorithms = ["watershed","kmeans", "pixel"]
for alg in segmentation_algorithms:
    dataset = ImageGraphDataset(img_path, labels, segmenter=alg)
    ds,s = stratified_split(dataset)
    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in train_loader:
        print(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
    # added an edge dim (2)
    model = EDGAT(4, 32, 2, 1, 2)
    # model = GCN(4,32,2,2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x = train(model, optimizer, train_loader)
    print(x)
    stats = test(model, train_loader)
    print(stats)