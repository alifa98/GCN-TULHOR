import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from collections import defaultdict


def build_spatial_graph(vocab_size, h3_edges, trajectory_sequences):
    """
    Builds a spatial-temporal graph combining:
    - H3 neighbor edges (spatial)
    - Sequential trajectory edges (temporal)

    Args:
        vocab_size (int): Total number of tokens (nodes)
        h3_edges (List[List[int]]): List of [from, to] edges from H3 neighbors
        trajectory_sequences (List[List[int]]): Original token index sequences from dataset

    Returns:
        networkx.Graph: Graph object with weighted edges
    """
    print("Building spatial graph...")
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(vocab_size))

    # Add H3 edges (spatial), default weight = 1
    G.add_edges_from(h3_edges, weight=1)

    # Add trajectory edges (temporal), accumulate frequency
    trajectory_edge_weights = count_consecutive_pairs(trajectory_sequences)
    for (u, v), weight in trajectory_edge_weights.items():
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight=weight)

    return G


def count_consecutive_pairs(sequences):
    """
    Count frequency of consecutive pairs in a list of sequences.

    Args:
        sequences (List[List[int]]): Sequences of token indices

    Returns:
        dict: Mapping of (i, j) -> frequency
    """
    pair_counts = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = tuple(sorted((seq[i], seq[i + 1])))
            pair_counts[pair] += 1
    return pair_counts


def normalize_adjacency(adj):
    """
    Normalizes adjacency matrix: A_hat = D^-1/2 * A * D^-1/2

    Args:
        adj (scipy.sparse.csr_matrix): Sparse adjacency matrix

    Returns:
        scipy.sparse.csr_matrix: Normalized adjacency matrix
    """
    print("Normalizing adjacency matrix...")
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # Symmetrize
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops

    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    return D_inv_sqrt @ adj @ D_inv_sqrt


def scipy_to_torch_sparse(scipy_mat):
    """
    Converts a scipy sparse matrix to PyTorch sparse tensor.

    Args:
        scipy_mat (scipy.sparse matrix): Input adjacency matrix

    Returns:
        torch.sparse.FloatTensor: PyTorch sparse tensor
    """
    print("Converting adjacency to PyTorch sparse tensor...")
    scipy_mat = sp.coo_matrix(scipy_mat)
    indices = np.vstack((scipy_mat.row, scipy_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(scipy_mat.data)
    shape = scipy_mat.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
