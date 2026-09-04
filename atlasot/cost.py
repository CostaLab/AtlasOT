from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix



def cosine_distance_tensor(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise cosine distance between two embedding matrices.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        First embedding matrix of shape (n_samples, n_features).
    Y : np.ndarray or torch.Tensor
        Second embedding matrix of shape (m_samples, n_features).

    Returns
    -------
    torch.Tensor
        Cosine distance matrix of shape (n_samples, m_samples), values in [0, 2].
    """
    X = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X.float()
    Y = torch.from_numpy(Y).float() if not isinstance(Y, torch.Tensor) else Y.float()
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    return 1 - torch.mm(X_norm, Y_norm.T)


def compute_knn_normalized_laplacian(
    spatial_coords: np.ndarray | torch.Tensor,
    n_neighbors: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Compute a normalized graph Laplacian from a k-NN Gaussian-affinity graph.

    Builds a k-nearest-neighbor graph over the spatial coordinates and weights
    each edge by a Gaussian kernel ``exp(-d^2 / (2 * sigma^2))``. Returns the
    symmetric normalized Laplacian (dense).

    Parameters
    ----------
    spatial_coords : np.ndarray or torch.Tensor
        Spatial coordinates of shape (n_spots, 2) or (n_spots, d).
    n_neighbors : int
        Number of nearest neighbors per spot (excluding itself).
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    torch.Tensor
        Normalized Laplacian matrix of shape (n_spots, n_spots).
    """
    from scipy.sparse import csgraph

    if isinstance(spatial_coords, torch.Tensor):
        spatial_coords = spatial_coords.numpy()
    X = np.asarray(spatial_coords, dtype=float)
    n = X.shape[0]
    if n < 2:
        return torch.zeros((n, n), dtype=torch.float32)

    # k-NN graph with Gaussian edge weights; query one extra neighbor so the
    # trivial self-neighbor (distance 0) can be dropped afterwards.
    k_total = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k_total).fit(X)
    dist, ind = nn.kneighbors(X)

    keep = ind != np.arange(n)[:, None]          # exclude self
    data = np.exp(-dist[keep] ** 2 / (2 * sigma ** 2))
    rows = np.repeat(np.arange(n), k_total)[keep.ravel()]
    cols = ind.ravel()[keep.ravel()]
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    # k-NN graph is not necessarily symmetric: symmetrize element-wise
    adjacency = adjacency.maximum(adjacency.T)

    L_scipy = csgraph.laplacian(adjacency, normed=True).toarray()
    return torch.tensor(L_scipy, dtype=torch.float32)


def compute_geodesic_distance(
    X: np.ndarray,
    k: int = 30,
    metric: str = 'cosine',
) -> np.ndarray:
    """Compute geodesic distance matrix from a k-NN graph of embeddings.

    Builds a k-nearest-neighbor graph on the embedding space, then computes
    all-pairs shortest-path distances. Automatically normalized to [0, 1].

    Parameters
    ----------
    X : np.ndarray
        Embedding matrix of shape (n_samples, n_features).
    k : int
        Number of neighbors for graph construction.
    metric : str
        Distance metric for k-NN ('cosine' or 'euclidean').

    Returns
    -------
    np.ndarray
        Normalized geodesic distance matrix of shape (n_samples, n_samples).
    """
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='distance', metric=metric, include_self=False)
    geo_dist = shortest_path(csgraph=knn_graph, directed=False)
    if np.isinf(geo_dist).any():
        finite_vals = geo_dist[np.isfinite(geo_dist)]
        max_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
        geo_dist[np.isinf(geo_dist)] = max_val * 1.5
    geo_dist /= geo_dist.max()
    return geo_dist


def compute_spatial_geodesic(
    coords: np.ndarray,
    features: np.ndarray,
    k_phys: int = 10,
    metric: str = 'cosine',
) -> tuple[np.ndarray, csr_matrix]:
    """Compute physically-constrained geodesic distance for spatial data.

    Edges are restricted to physical neighbors (k_phys), while edge weights
    are determined by expression feature similarity. Also returns the affinity
    matrix for downstream graph-based smoothing.

    Parameters
    ----------
    coords : np.ndarray
        Physical coordinates of shape (n_spots, 2) or (n_spots, d).
    features : np.ndarray
        Expression features of shape (n_spots, n_features).
    k_phys : int
        Number of physical neighbors to consider.
    metric : str
        Distance metric for feature similarity ('cosine' or 'euclidean').

    Returns
    -------
    geo_matrix : np.ndarray
        Normalized geodesic distance matrix of shape (n_spots, n_spots).
    adj_aff : csr_matrix
        Sparse affinity matrix for graph smoothing.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    n_samples = features.shape[0]

    # 1. Physical space neighbor construction (define which points can be connected)
    # Set k_phys slightly larger (e.g., 10-15) to ensure the graph is connected and avoid disconnected components due to individual points
    nbrs_phys = NearestNeighbors(n_neighbors=k_phys, metric='euclidean').fit(coords)
    phys_dists, phys_indices = nbrs_phys.kneighbors(coords)

    # 2. Construct edge weights for sparse graph (determined by feature similarity)
    # We traverse physical neighbors and calculate their distance in feature space
    row = []
    col = []
    data_dist = []
    data_aff = []

    # Normalize features for distance calculation
    if metric == 'cosine':
        # Manual L2 normalization for easier dot product calculation
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        features_norm = features / norms

    for i in range(n_samples):
        # Get physical neighbor indices for the i-th point (skip self, 0 is usually self)
        neighbors = phys_indices[i, 1:]
        
        if metric == 'cosine':
            # Cosine Dist = 1 - (A . B)
            # Distance range is [0, 2]
            # More similar means smaller distance
            dists = 1 - np.dot(features_norm[neighbors], features_norm[i])
            # Fix possible negative value precision errors
            dists = np.maximum(dists, 0)
            
            # Affinity: exp(-dist)
            affs = np.exp(-dists)
        else:
            # Euclidean
            dists = np.linalg.norm(features[neighbors] - features[i], axis=1)
            
            # Gaussian Kernel
            # Reduce sigma (x0.5) to make affinity "sharper", reducing SSIM drop caused by oversmoothing
            sigma = np.mean(dists) * 0.5 + 1e-10
            affs = np.exp(-dists**2 / (2 * sigma**2))

        # Key point: We only establish edges if physically adjacent, edge length is expression difference
        row.extend([i] * len(neighbors))
        col.extend(neighbors)
        data_dist.extend(dists)
        data_aff.extend(affs)

    # 3. Construct sparse adjacency matrix (Distance) - for Geodesic calculation
    adj_dist = csr_matrix((data_dist, (row, col)), shape=(n_samples, n_samples))
    adj_dist = adj_dist + adj_dist.T
    
    # 4. Construct sparse adjacency matrix (Affinity) - for post-processing smoothing
    adj_aff = csr_matrix((data_aff, (row, col)), shape=(n_samples, n_samples))
    adj_aff = adj_aff + adj_aff.T
    
    # 5. Calculate All-Pairs Shortest Path (Geodesic Distance)
    # This calculates "physical travel distance considering tissue boundaries"
    geo_matrix = shortest_path(adj_dist, directed=False)

    # Handle inf values caused by disconnected graph
    if np.isinf(geo_matrix).any():
        finite_vals = geo_matrix[np.isfinite(geo_matrix)]
        max_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
        geo_matrix[np.isinf(geo_matrix)] = max_val * 1.5

    geo_matrix /= geo_matrix.max()
    return geo_matrix, adj_aff



def graph_smooth_results(
    features: np.ndarray,
    adj_matrix: csr_matrix,
    alpha: float = 0.5,
    n_iter: int = 2,
) -> np.ndarray:
    """Smooth imputed gene expression via random walk on spatial graph.

    Applies iterative smoothing: F_{t+1} = (1-α)F₀ + α·P·F_t,
    where P is the row-normalized adjacency matrix.

    Parameters
    ----------
    features : np.ndarray
        Imputed gene expression of shape (n_spots, n_genes).
    adj_matrix : csr_matrix
        Sparse spatial adjacency/affinity matrix.
    alpha : float
        Propagation strength (0 = no smoothing, 1 = full smoothing).
    n_iter : int
        Number of smoothing iterations.

    Returns
    -------
    np.ndarray
        Smoothed expression matrix of shape (n_spots, n_genes).
    """
    from scipy.sparse import diags
    
    # Normalize adjacency matrix P = D^-1 * A
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero
    D_inv = diags(1.0 / degrees)
    P = D_inv.dot(adj_matrix)
    
    # Iterative smoothing: F_{t+1} = (1-alpha)F_0 + alpha * P * F_t
    F = features.copy()
    F0 = features.copy()
    
    for _ in range(n_iter):
        F = (1 - alpha) * F0 + alpha * P.dot(F)
        
    return F
