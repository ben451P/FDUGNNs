import numpy as np
import torch.nn.functional as F
import torch_directml


def train(model, optimizer, train_dataloader):
    device = torch_directml.device()
    device = device if device else "cpu"
    for epoch in range(5):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = batch.to(device)
            if batch.x is None:
                print("Skipping batch due to missing node features")
                continue

            print(batch[0].edge_attr, batch[0].edge_index, batch[0].y, batch[0].x)
            arr = batch[0].edge_index
            print(np.count_nonzero((arr != 0) & (arr != 143)))
            print(arr)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(out, batch.y)  # reshaping
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    return model, optimizer, total_loss / len(train_dataloader)
