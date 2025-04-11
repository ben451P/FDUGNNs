import numpy as np
import torch
from scipy.spatial import Delaunay
from skimage.measure import regionprops
import networkx as nx


# Create a graph from superpixels using Delaunay triangulation
def delauny_graph_from_superpixels(segments, image):
    # Extract centroids of regions
    props = regionprops(segments + 1)
    centroids = np.array([prop.centroid for prop in props])

    if len(centroids) < 3:
        return None

    # Perform Delaunay triangulation to create edges
    tri = Delaunay(centroids)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))

    G = nx.Graph(list(edges))

    # Assign node features (RGB mean for each segment)
    for i, prop in enumerate(props):
        mask = segments == i
        mean_color = np.mean(image[mask], axis=0)
        G.nodes[i]["x"] = torch.tensor(mean_color, dtype=torch.float)

    # Assign edge attributes (Euclidean distance between centroids)
    for edge in G.edges:
        i, j = edge
        dist = np.linalg.norm(centroids[i] - centroids[j])
        G.edges[edge]["edge_attr"] = torch.tensor([dist], dtype=torch.float)

    return G
