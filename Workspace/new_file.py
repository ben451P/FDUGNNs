import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
import networkx as nx
import numpy as np
from skimage.segmentation import find_boundaries
from preprocessing import create_superpixels, superpixel_properties

import networkx as nx


import numpy as np # linear algebra

import networkx as nx
import torch


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
        node_feat = torch.tensor([node_data[i] for i in node_data],dtype=torch.float)
        G.add_node(label, x=node_feat)
    
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
                    edge_feat = torch.tensor([d_mean, d_med,feats['centroid_dist'],feats['boundary_len']],dtype=torch.float)
                    G.add_edge(l1, l2,
                            edge_attr=edge_feat)
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
            G.nodes[n]['x'] = torch.cat((G.nodes[n]['x'],torch.tensor([texture[n]])))
        if n in shape:
            G.nodes[n]["x"] = torch.cat((G.nodes[n]['x'],torch.tensor([shape[n][i] for i in shape[n]])))
    G.nodes[n]["x"].to(torch.float)
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

def compute_texture(img, segments, P=8, R=1):
    """
    Compute the mean LBP value per superpixel.
    - P: number of circularly symmetric neighbour set points.
    - R: radius.
    Returns a dict: {label: mean_lbp}.
    """
    gray = rgb2gray(img) *255
    # print(gray, gray.shape)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    texture = {}
    for label in np.unique(segments):
        mask = segments == label
        texture[label] = lbp[mask].mean()
    return texture


def compute_shape_metrics(segments):
    """
    For each labeled region, compute:
      - area (pixel count)
      - perimeter
      - compactness = perimeter^2 / area
    Returns dict of dicts: {label: {'area':…, 'perimeter':…, 'compactness':…}}.
    """
    props = regionprops(segments.astype(int))
    shape = {}
    for p in props:
        label = p.label
        area = p.area
        perim = p.perimeter
        compact = (perim ** 2) / area if area > 0 else 0
        shape[label] = {
            'area': area,
            'perimeter': perim,
            'compactness': compact
        }
    return shape
