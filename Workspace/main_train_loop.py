import torch
import torch.optim as optim
import pandas as pd
import os
from torch_geometric.loader import DataLoader
from mainloop_functions import train, test
from models.DGAT import DynamicGAT
from models.EDGAT import DynamicEdgeGAT
from models.GAT import StaticGAT

# Load saved datasets
train_path = r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\saved_datasets\delauny_graph\train_final.pt"
test_path  = r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\saved_datasets\delauny_graph\test_final.pt"

train_loader = torch.load(train_path, weights_only=False)
test_loader = torch.load(test_path, weights_only=False)

sample_batch = next(iter(train_loader))
sample = sample_batch[0] if isinstance(sample_batch, list) else sample_batch
in_dim = sample.x.shape[1]
hidden_dim = 64
out_dim = len(torch.unique(torch.cat([s[1].unsqueeze(0) if isinstance(s, tuple) else s.y.unsqueeze(0) for s in sample_batch])))

# Prepare models
models = {
    "dgat": DynamicGAT(in_dim, hidden_dim, out_dim),
    "edgat": DynamicEdgeGAT(in_dim, hidden_dim, out_dim),
    "gat": StaticGAT(in_dim, hidden_dim, out_dim)
}

results = []

for name, model in models.items():
    print(f"\n=== Training {name.upper()} ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, _ = train(model, optimizer, train_loader)

    print(f"\n=== Testing {name.upper()} ===")
    metrics = test(model, test_loader)

    # Save model
    model_save_path = f"saved_models/{name}.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # Append results
    metrics["model"] = name
    results.append(metrics)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("summary.csv", index=False)
print("\nResults saved to results/summary.csv")