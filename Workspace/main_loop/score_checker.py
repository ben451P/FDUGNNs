import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ROC
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
from image_preprocessing.standard import image_to_graph
import torch_directml

from models.EDGAT import DynamicEdgeGAT


class ImageGraphDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.data = []
        for path, label in zip(image_paths, labels):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            try:
                graph = image_to_graph(image)
            except Exception as e:
                print(e)
                continue

            if graph is None:
                continue

            graph.y = torch.tensor(label, dtype=torch.long)
            self.data.append(graph)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ScoreChecker:
    @staticmethod
    def main(model, test_dataloader):
        device = torch_directml.device()
        device = device if device else "cpu"

        accuracy = Accuracy(task="binary")
        precision = Precision(task="binary")
        recall = Recall(task="binary")
        f1score = F1Score(task="binary")
        ROC = ROC(task="binary")

        metrics = {
            "acc": [0, accuracy],
            "pre": [0, precision],
            "recall": [0, recall],
            "f1": [0, f1score],
        }

        model.eval()
        roc = 0
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                if batch.x is None:
                    print("Skipping batch due to missing node features")
                    continue

                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = F.cross_entropy(out, batch.y)
                total_loss += loss.item()

                preds = out.argmax(dim=1)
                preds_prob = out.softmax(dim=1)
                fpr, tpr, _ = ROC(
                    preds_prob, F.one_hot(batch.y, num_classes=2)
                )  # ROC needs probabilities
                roc += torch.trapz(tpr, fpr)
                for key in metrics.keys():
                    metrics[key][0] += metrics[key][1](preds, batch.y)
                total_samples += batch.y.size(0)

        for key in metrics:
            metrics[key][0] /= len(test_dataloader)
        roc /= len(test_dataloader)
        avg_loss = total_loss / len(test_dataloader)
        print(f"{avg_loss:.4f}")
        print(f"ROC: {roc}")
        for key in metrics:
            print(f"{key}: {metrics[key][0]}")

        metrics["loss"] = avg_loss
        metrics["ROC"] = roc
        return metrics
