import networkx as nx
import numpy as np
from preprocessing import create_superpixels, superpixel_properties
from skimage.segmentation import find_boundaries
from preprocessing import compute_shape_metrics, compute_texture


def create_graph(img, method='slic', **seg_kwargs):
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
                    G.add_edge(l1, l2, diff_mean=d_mean, diff_median=d_med)
    return G


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
