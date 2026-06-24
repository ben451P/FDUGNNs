import os
import torch
import torch.optim as optim
import pandas as pd
from torch_geometric.loader import DataLoader

from mainloop_functions  import train, test
from models.DGAT import DynamicGAT as DGAT
from models.EDGAT import DynamicEdgeGAT as EDGAT
from models.GAT import StaticGAT as GAT
from models.GCN import GCN
from dataset_definition import ImageGraphDataset
from skimage import io
import random
import matplotlib.pyplot as plt
from preprocessing import stratified_split
from torch.utils.data import random_split
random.seed(0)

# Increment based on existing files
run = "13"


# Load your image paths and labels
img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\benign'
image_paths = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)]

img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\malignant'
image_paths2 = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)]
labels2 = [1] * len(image_paths2)

### Uncomment to add synthetic data ###
# img_dir = r'C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\image_dataset\generated_malignant'
# image_paths3 = [io.imread(os.path.join(img_dir, f)) for f in os.listdir(img_dir)]
# labels3 = [1] * len(image_paths3)

### Uncomment to balance dataset 1:1 ratio ###
# image_paths = random.sample(image_paths, (len(image_paths2) + len(image_paths3) * 1))
labels = [0] * len(image_paths)

# Combine real data
real_images = image_paths + image_paths2
real_labels = labels + labels2

### Uncomment to add synthetic data ###
# all_train_images = real_images + image_paths3
# all_train_labels = real_labels + labels3

full_real_dataset = ImageGraphDataset(real_images, real_labels, segmenter="slic")
train_size = int(len(full_real_dataset) * 0.8)
val_size = len(full_real_dataset) - train_size
train_ds, val_ds = random_split(full_real_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(val_ds, batch_size=4, shuffle=False)


in_dim, out_dim = 4, 2
hid_dim = 64

dict_models = {
    "dgat": DGAT(in_dim, hid_dim, out_dim, edge_dim=2),
    "edgat": EDGAT(in_dim, hid_dim, out_dim, edge_dim=2),
    "gat": GAT(in_dim, hid_dim, out_dim, edge_dim=2),
    "gcn": GCN(in_dim, hid_dim, out_dim)
}

results = []
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
    plt.savefig(f"results/{name}_training_loss_plot{run}.png")  # Saves locally
    plt.clf()

    torch.save(model.state_dict(), f"saved_models/{name}{run}.pt")
    stats['model'] = name
    results.append(stats)

# write summary
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv(f"results/summary{run}.csv", index=False)
print("All done.")
