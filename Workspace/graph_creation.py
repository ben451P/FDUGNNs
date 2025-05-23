import networkx as nx
import numpy as np
from preprocessing import create_superpixels, superpixel_properties
from skimage.segmentation import find_boundaries
from preprocessing import compute_shape_metrics, compute_texture


import networkx as nx
from torch_geometric.utils import from_networkx
import torch


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from skimage import io, color
from skimage.segmentation import quickshift, mark_boundaries, slic, felzenszwalb, watershed
from skimage.color import label2rgb
import networkx as nx
import torch
import cv2

from scipy.spatial import Delaunay
from skimage.measure import regionprops
# Manually construct Region Adjacency Graph (RAG)
def construct_rag(segments):
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

# Create a graph from superpixels using Delaunay triangulation
def create_graph_from_superpixels(segments, image):
    # Extract centroids of regions
    props = regionprops(segments + 1)
    centroids = np.array([prop.centroid for prop in props])

    # Check if we have enough regions to form a graph
    if len(centroids) < 3:
        return None  # Avoid issues with Delaunay requiring at least 3 points

    # Perform Delaunay triangulation to create edges
    tri = Delaunay(centroids)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))

    # Create a graph
    G = nx.Graph(list(edges))

    # Assign node features (RGB mean for each segment)
    for i, prop in enumerate(props):
        mask = segments == i
        mean_color = np.mean(image[mask], axis=0)  # Extract mean color
        G.nodes[i]['x'] = torch.tensor(mean_color, dtype=torch.float)

    # Assign edge attributes (Euclidean distance between centroids)
    for edge in G.edges:
        i, j = edge
        dist = np.linalg.norm(centroids[i] - centroids[j])
        G.edges[edge]['edge_attr'] = torch.tensor([dist], dtype=torch.float)
    return G

def create_graph(img, method='slic', **seg_kwargs):
    """Builds a RAG: nodes carry superpixel stats; edges carry pairwise differences."""
    segments = create_superpixels(img, method, **seg_kwargs)
    props, coords = superpixel_properties(img, segments)

    G = nx.Graph()
    # add nodes
    for label, p in props.items():
        mean = p['mean']
        median = p['median']
        centroid_y = coords[label]['centroid_y']
        centroid_x = coords[label]['centroid_x']

        # Stack node attributes into one tensor for PyG
        node_feat = torch.tensor([mean, median, centroid_y, centroid_x], dtype=torch.float)
        G.add_node(label, x=node_feat)

    # add edges between spatially adjacent superpixels
    h, w = segments.shape
    for y in range(h - 1):
        for x in range(w - 1):
            l1 = segments[y, x]
            for dy, dx in [(1, 0), (0, 1)]:
                l2 = segments[y + dy, x + dx]
                if l1 != l2 and not G.has_edge(l1, l2):
                    d_mean = abs(props[l1]['mean'] - props[l2]['mean'])
                    d_med  = abs(props[l1]['median'] - props[l2]['median'])
                    edge_feat = torch.tensor([d_mean, d_med], dtype=torch.float)
                    G.add_edge(l1, l2, edge_attr=edge_feat)

    # convert to PyG Data object, preserving node and edge attributes
    return G

# def create_graph(img, method='slic', **seg_kwargs):
#     """Builds a RAG: nodes carry superpixel stats; edges carry pairwise differences."""
#     segments = create_superpixels(img, method, **seg_kwargs)
#     props, coords = superpixel_properties(img, segments)

#     G = nx.Graph()
#     # add nodes
#     for label, p in props.items():
#         node_data = {
#             'mean': p['mean'],
#             'median': p['median'],
#             'centroid_y': coords[label]['centroid_y'],
#             'centroid_x': coords[label]['centroid_x'],
#         }
#         G.add_node(label, **node_data)

#     # add edges between spatially adjacent superpixels
#     h, w = segments.shape
#     for y in range(h - 1):
#         for x in range(w - 1):
#             l1 = segments[y, x]
#             for dy, dx in [(1, 0), (0, 1)]:
#                 l2 = segments[y + dy, x + dx]
#                 if l1 != l2 and not G.has_edge(l1, l2):
#                     d_mean = abs(props[l1]['mean'] - props[l2]['mean'])
#                     d_med  = abs(props[l1]['median'] - props[l2]['median'])
#                     G.add_edge(l1, l2, diff_mean=d_mean, diff_median=d_med)
#     return G


def create_graph_more_features(img, method='slic', **seg_kwargs):
    """Builds a RAG: nodes carry superpixel stats; edges carry pairwise differences."""
    segments = create_superpixels(img, method, **seg_kwargs)
    props, coords = superpixel_properties(img, segments)

    G = nx.Graph()
    # add nodes
    for label, p in props.items():
        node_data = {
            'mean': p['mean'],
            'median': p['median'],
            'centroid_y': coords[label]['centroid_y'],
            'centroid_x': coords[label]['centroid_x'],
        }
        G.add_node(label, **node_data)
    
    # after props, coords computed:
    texture = compute_texture(img, segments)
    shape   = compute_shape_metrics(segments)
    G = augment_node_features(G, texture, shape)


    # add edges between spatially adjacent superpixels
    h, w = segments.shape
    for y in range(h - 1):
        for x in range(w - 1):
            l1 = segments[y, x]
            for dy, dx in [(1, 0), (0, 1)]:
                l2 = segments[y + dy, x + dx]
                if l1 != l2 and not G.has_edge(l1, l2):
                    d_mean = abs(props[l1]['mean'] - props[l2]['mean'])
                    d_med  = abs(props[l1]['median'] - props[l2]['median'])
                    edge_feats = compute_edge_features(img, segments, props, coords)
                    feats = edge_feats[(min(l1, l2), max(l1, l2))]
                    G.add_edge(l1, l2,
                            diff_mean=d_mean,
                            diff_median=d_med,
                            centroid_dist=feats['centroid_dist'],
                            boundary_len=feats['boundary_len'])
    return G


def augment_node_features(G, texture, shape):
    """
    Given:
      - G: the RAG graph with existing 'mean','median','centroid_x/y'
      - texture: {label: mean_lbp}
      - shape: {label: {area, perimeter, compactness}}
    Adds these as node attributes.
    """
    for n in G.nodes:
        if n in texture:
            G.nodes[n]['lbp_mean'] = texture[n]
        if n in shape:
            G.nodes[n].update(shape[n])
    return G


def compute_edge_features(img, segments, props, coords):
    """
    Walk adjacency and, for each edge u–v, compute:
      - centroid distance
      - shared boundary length
    Return dict keyed by (u,v): {'centroid_dist':…, 'boundary_len':…}.
    """
    edge_feats = {}
    # Precompute boundary map
    bd = find_boundaries(segments, mode='thick')
    # For each adjacent pair, as in create_graph:
    h, w = segments.shape
    for y in range(h-1):
        for x in range(w-1):
            u = segments[y, x]
            for dy, dx in [(1,0),(0,1)]:
                v = segments[y+dy, x+dx]
                if u != v:
                    key = tuple(sorted((u, v)))
                    if key not in edge_feats:
                        # centroid euclidean distance
                        cy1, cx1 = coords[u]['centroid_y'], coords[u]['centroid_x']
                        cy2, cx2 = coords[v]['centroid_y'], coords[v]['centroid_x']
                        dist = ((cy1-cy2)**2 + (cx1-cx2)**2) ** 0.5
                        # boundary length approx: count shared boundary pixels
                        mask = (segments == u) & np.roll(segments, shift=-dy, axis=0) * (segments == v)
                        # sum boundary in both primary directions
                        b_len = ((bd & ((segments == u) | (segments == v))).sum())
                        edge_feats[key] = {
                            'centroid_dist': dist,
                            'boundary_len': int(b_len)
                        }
    return edge_feats
