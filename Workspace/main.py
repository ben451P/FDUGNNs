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
import random
import matplotlib.pyplot as plt

random.seed(0)

# Simple collate to batch variable-size graphs
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

# Load your image paths and labels
img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
image_paths = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)]
labels = [0] * len(image_paths)
cutoff = len(labels)

img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\malignant'
image_paths2 = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)]
label2 = [1] * len(image_paths2)

use = random.sample(image_paths,len(image_paths2))
use_labels = [0] * len(image_paths2)
use += image_paths2
use_labels += [1] * len(image_paths2)


# Prepare dataset and loaders
dataset = ImageGraphDataset(use, use_labels, segmenter='slic', n_segments=100)
train_ds, test_ds = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=graph_collate)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, collate_fn=graph_collate)

# Infer feature and class sizes from first batch

in_dim, out_dim = 4, 2
hid_dim = 64  # you can tune this

# Initialize models dynamically
dict_models = {
    "dgat": DGAT(in_dim, hid_dim, out_dim, edge_dim=2),
    "edgat": EDGAT(in_dim, hid_dim, out_dim, edge_dim=2),
    "gat": GAT(in_dim, hid_dim, out_dim, edge_dim=2)
}
results = []

# Training and evaluation loop
os.makedirs("saved_models", exist_ok=True)
for name, model in dict_models.items():
    print(f"--- {name.upper()} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, best_loss, loss_over_time = train(model, optimizer, train_loader)
    stats = test(model, test_loader)
    print(best_loss, loss_over_time)
    losses = loss_over_time
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{name}_training_loss_plot.png")  # Saves locally
    plt.clf()
    # Save
    torch.save(model.state_dict(), f"saved_models/{name}3.pt")
    stats['model'] = name
    results.append(stats)

# Save summary
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/summary3.csv", index=False)
print("All done.")
# print(dataset.data, dataset.labels)