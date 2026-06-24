from train.preprocessing import create_superpixels, superpixel_properties

import networkx as nx
import torch

def create_graph(img, method='slic', **seg_kwargs):
    """Builds a RAG: nodes carry superpixel stats; edges carry pairwise differences."""
    segments = create_superpixels(img, method, **seg_kwargs)
    props, coords = superpixel_properties(img, segments)

    G = nx.Graph()
    # add superpixels as nodes
    for label, p in props.items():
        mean = p['mean']
        median = p['median']
        centroid_y = coords[label]['centroid_y']
        centroid_x = coords[label]['centroid_x']

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

    return G
