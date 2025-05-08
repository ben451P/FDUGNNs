import networkx as nx
import numpy as np
import torch
from scipy.spatial import Delaunay
from skimage.measure import regionprops

def construct_rag(segments):
    """Manual Region Adjacency Graph construction"""
    G = nx.Graph()
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            node = segments[i, j]
            G.add_node(node)
            if i > 0 and segments[i - 1, j] != node:
                G.add_edge(node, segments[i - 1, j])
            if j > 0 and segments[i, j - 1] != node:
                G.add_edge(node, segments[i, j - 1])
    return G

def delaunay_graph_from_superpixels(segments, image):
    """Delaunay triangulation-based graph construction"""
    props = regionprops(segments + 1)
    centroids = np.array([prop.centroid for prop in props])

    if len(centroids) < 3:
        return None

    tri = Delaunay(centroids)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))

    G = nx.Graph(list(edges))

    # Node features
    for i, prop in enumerate(props):
        mask = segments == i
        mean_color = np.mean(image[mask], axis=0)
        G.nodes[i]["x"] = torch.tensor(mean_color, dtype=torch.float)

    # Edge attributes
    for edge in G.edges:
        i, j = edge
        dist = np.linalg.norm(centroids[i] - centroids[j])
        G.edges[edge]["edge_attr"] = torch.tensor([dist], dtype=torch.float)

    return G