
from skopt import BayesSearchCV
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
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.base import BaseEstimator
import torch
import torch.nn.functional as F

class SklearnTorchWrapper(BaseEstimator):
    def __init__(self, input_dim=2, hidden_dim=32, lr=0.01, weight_decay=0.0, epochs=10,
                 train_loader=None, val_loader=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader

    def build_model(self):
        return GAT(4, self.hidden_dim, 2, 1, 2)  # Customize for your real GAT

    def fit(self, X=None, y=None):
        self.model = self.build_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.losses = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                print(batch)
                batch = batch.to(device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            self.losses.append(avg_loss)
        return self

    def predict(self, loader=None):
        self.model.eval()
        preds = []
        device = next(self.model.parameters()).device
        loader = loader if loader is not None else self.val_loader
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=1).cpu()
                preds.append(pred)
        return torch.cat(preds).numpy()

    def score(self, X=None, y=None):
        # X, y are ignored; we use val_loader
        preds = self.predict()
        true = []
        for batch in self.val_loader:
            true.append(batch.y.cpu())
        true = torch.cat(true).numpy()

        return (preds == true).mean()





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
search = BayesSearchCV(
    SklearnTorchWrapper(train_loader=train_loader, val_loader=train_loader),
    {
        'hidden_dim': (16, 128),
        'lr': (1e-4, 1e-1, 'log-uniform'),
        'weight_decay': (1e-5, 1e-2, 'log-uniform'),
        'epochs': (5, 20),
    },
    n_iter=10,
    cv=3,
    scoring='accuracy',
    verbose=0
)

# Sample data
X = np.random.rand(500, 10)
y = np.random.randint(0, 2, 500)

search.fit(X, y)

print("Best score:", search.best_score_)
print("Best params:", search.best_params_)

import matplotlib.pyplot as plt

losses = search.best_estimator_.losses
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.title("Training Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_plot.png")  # Saves locally
plt.show()
