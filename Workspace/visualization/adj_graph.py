import numpy
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic, watershed, felzenszwalb
from skimage.color import label2rgb
import networkx as nx
import cv2
import os
import random

from train.graph_creation import create_graph

# image = io.imread('pdfUploaded.jpg')

random.seed(1)
root = "/Users/benlozzano/VS-Code-Coding/Completed/FDU Internship Final/image_dataset/malignant/"
images = os.listdir(root)

image = cv2.imread(os.path.join(root, random.sample(images,1)[0]))



# Perform superpixel segmentation using SLIC
segments = slic(image, n_segments=100)

# Visualize the segmented image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[1].imshow(mark_boundaries(image, segments))
ax[1].set_title("SLIC Segmentation")
ax[1].axis('off')


# Create the graph
G = create_graph(image, "slic", n_segments=100)

# Visualize the superpixel segments with boundaries
ax[0].imshow(image)
ax[0].set_title("Regular Image")
ax[0].axis('off')

plt.tight_layout()
plt.show()

# Visualize the graph using NetworkX
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)  # Layout for visualization
nx.draw(G, pos, with_labels=False, node_size=20, node_color="skyblue", font_size=8, edge_color="gray")
plt.title("Superpixel Region Adjacency Graph (RAG) Visualization")
plt.show()