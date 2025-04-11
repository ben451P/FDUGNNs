import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from image_preprocessing.standard import image_to_graph


class ImageGraphDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.data = []
        for path, label in zip(image_paths, labels):
            # In ImageGraphDataset class:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image / 255.0).astype(np.float32)  # Cast to float32
            # Apply thresholding and segmentation here
            try:
                graph = image_to_graph(image)  # Calls the modified function
            # ... rest remains unchanged ...
            except Exception as e:
                print(e)
                continue

            if graph is None:
                print("Skipping")
                continue
            print(1)
            graph.y = torch.tensor(label, dtype=torch.long)
            self.data.append(graph)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
