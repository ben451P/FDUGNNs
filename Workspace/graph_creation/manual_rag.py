# Manually construct Region Adjacency Graph (RAG)
import networkx as nx


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
