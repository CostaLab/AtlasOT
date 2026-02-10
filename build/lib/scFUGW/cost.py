import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def umap_cost(
    rna,
    key_added: str = "UMAP4Distance",
    n_components: int = 2,
    l2_normalize: bool = True,
    store_in_obsp: bool = True,
    obsp_key: str = "cost_matrix",
    dtype: torch.dtype = torch.float32,
):
    """
    依据 RNA 表达矩阵的 UMAP 嵌入计算成对距离（代价矩阵）。

    参数:
        rna: AnnData 对象，包含 RNA 表达矩阵在 rna.X
        key_added: 保存 UMAP 嵌入到 rna.obsm 的键名
        n_components: UMAP 维度
        l2_normalize: 是否对 UMAP 行向量做 L2 归一化
        store_in_obsp: 是否将距离矩阵存入 rna.obsp
        obsp_key: 距离矩阵在 rna.obsp 中的键名
        dtype: 计算所用 torch dtype

    返回:
        torch.Tensor 形状为 (n_cells, n_cells) 的成对距离矩阵
    """
    # 邻接与 UMAP 嵌入
    sc.pp.neighbors(rna)
    sc.tl.umap(rna, key_added=key_added, n_components=n_components)

    # 取出嵌入
    embedding = rna.obsm[key_added]
    if not isinstance(embedding, np.ndarray):
        embedding = np.asarray(embedding)

    # 可选 L2 归一化（逐行）
    if l2_normalize:
        tensor_emb = torch.from_numpy(embedding).to(dtype)
        norms = torch.linalg.norm(tensor_emb, dim=1).reshape(-1, 1)
        # 避免除以 0
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        tensor_emb = tensor_emb / norms
    else:
        tensor_emb = torch.from_numpy(embedding).to(dtype)

    # 成对距离（欧氏距离）
    cost_matrix = torch.cdist(tensor_emb, tensor_emb)

    if store_in_obsp:
        # 与现有代码保持一致，存 torch.Tensor；如需 numpy 可改为 cost_matrix.cpu().numpy()
        rna.obsp[obsp_key] = cost_matrix

    return cost_matrix





# def cosine_distance_tensor(X, Y):
#     import torch.nn.functional as F

#     # 转成 tensor 并确保 float32
#     X = torch.from_numpy(X).float()
#     Y = torch.from_numpy(Y).float()

#     # L2 归一化
#     X_norm = F.normalize(X, p=2, dim=1)
#     Y_norm = F.normalize(Y, p=2, dim=1)

#     # 计算余弦相似度矩阵
#     sim_matrix = torch.mm(X_norm, Y_norm.T)

#     weight_matrix = torch.exp(sim_matrix)
#     target_weights = torch.sum(weight_matrix, dim=0)
#     source_weights = torch.sum(weight_matrix, dim=1)
#     target_weights = target_weights / torch.sum(target_weights)
#     source_weights = source_weights / torch.sum(source_weights)

#     # 余弦距离 = 1 - 余弦相似度
#     dist_matrix = 1 - sim_matrix

#     return dist_matrix, target_weights, source_weights  # torch.Tensor

def cosine_distance_tensor(X, Y):
    X = torch.from_numpy(X).float() if not isinstance(X, torch.Tensor) else X.float()
    Y = torch.from_numpy(Y).float() if not isinstance(Y, torch.Tensor) else Y.float()
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    return 1 - torch.mm(X_norm, Y_norm.T)


def knn_cosine_distance(
    adata,
    obsm_key: str = "RNA_pca_l2_norm",
    n_neighbors: int = 20,
    fill_value: float = 1.0,
):
    """
    基于 AnnData 的某个嵌入（位于 obsm[obsm_key]），使用 KNN + 余弦度量
    计算稀疏样式的距离矩阵（非邻居位置填充 fill_value，默认 1.0），对角为 0。

    参数:
        adata: AnnData 对象
        obsm_key: 使用的嵌入键名（如 'RNA_pca_l2_norm'）
        n_neighbors: KNN 中的邻居数 k
        fill_value: 非邻域默认填充值（余弦距离的上界为 1）

    返回:
        np.ndarray，形状为 (n_cells, n_cells)
    """
    X = adata.obsm[obsm_key]
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)

    n_spots = X.shape[0]
    dist_mat = np.full((n_spots, n_spots), fill_value, dtype=np.float32)
    for i in range(n_spots):
        for j in range(indices.shape[1]):
            neighbor_idx = indices[i, j]
            dist = distances[i, j]
            dist_mat[i, neighbor_idx] = dist

    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def compute_knn_normalized_laplacian(spatial_coords, n_neighbors: int = 5, sigma: float = 1.0):
    """
    基于空间坐标使用高斯核图（由欧氏距离生成）并计算归一化拉普拉斯矩阵。

    参数:
        spatial_coords: 空间坐标数组，形状为 (n_samples, 2) 或 (n_samples, d)
        n_neighbors: 预留参数（如需改为基于KNN稀疏图可使用），当前实现未使用
        sigma: 高斯核带宽

    返回:
        torch.Tensor，归一化拉普拉斯矩阵 (float32)
    """
    from scipy.sparse import csgraph
    from scipy.spatial.distance import cdist

    if isinstance(spatial_coords, torch.Tensor):
        spatial_coords = spatial_coords.numpy()

    dist_matrix = cdist(spatial_coords, spatial_coords, metric='euclidean')
    adjacency_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(adjacency_matrix, 0)

    L_scipy = csgraph.laplacian(adjacency_matrix, normed=True)
    L_torch = torch.tensor(L_scipy, dtype=torch.float32)
    return L_torch


def compute_geodesic_distance(X, k=30):
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    geo_dist = shortest_path(csgraph=knn_graph, directed=False)
    if np.isinf(geo_dist).any():
        finite_vals = geo_dist[np.isfinite(geo_dist)]
        max_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
        geo_dist[np.isinf(geo_dist)] = max_val * 1.5
    return geo_dist


def compute_spatial_geodesic(coords, features, k_phys=10, metric='cosine'):
    """
    构建一个物理约束的图，但边权重由表达特征决定，然后计算测地线距离。
    修改：同时返回用于平滑的亲和度矩阵。
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    n_samples = features.shape[0]

    # 1. 物理空间的邻居构建 (定义哪些点可以相连)
    # k_phys 稍微设大一点 (如 10-15)，确保图是连通的，不会因为个别点导致断路
    nbrs_phys = NearestNeighbors(n_neighbors=k_phys, metric='euclidean').fit(coords)
    phys_dists, phys_indices = nbrs_phys.kneighbors(coords)

    # 2. 构建稀疏图的边权重 (由特征相似度决定)
    # 我们遍历物理近邻，计算它们在特征空间中的距离
    row = []
    col = []
    data_dist = []
    data_aff = []

    # 归一化特征以便计算距离
    if metric == 'cosine':
        # 手动 L2 归一化，方便算点积
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # 避免除零
        norms[norms == 0] = 1e-10
        features_norm = features / norms

    for i in range(n_samples):
        # 获取第 i 个点的物理邻居索引 (跳过自身, 0通常是自身)
        neighbors = phys_indices[i, 1:]
        
        if metric == 'cosine':
            # Cosine Dist = 1 - (A . B)
            # 这里的距离范围是 [0, 2]
            # 越相似，距离越小
            dists = 1 - np.dot(features_norm[neighbors], features_norm[i])
            # 修正可能的负值精度误差
            dists = np.maximum(dists, 0)
            
            # Affinity: exp(-dist)
            affs = np.exp(-dists)
        else:
            # Euclidean
            dists = np.linalg.norm(features[neighbors] - features[i], axis=1)
            
            # Gaussian Kernel
            # 减小 sigma (x0.5)，使亲和度更“锐利”，减少过度平滑导致的 SSIM 下降
            sigma = np.mean(dists) * 0.5 + 1e-10
            affs = np.exp(-dists**2 / (2 * sigma**2))

        # 这里的关键：如果物理相邻，我们才建立边，边的长度是表达差异
        row.extend([i] * len(neighbors))
        col.extend(neighbors)
        data_dist.extend(dists)
        data_aff.extend(affs)

    # 3. 构建稀疏邻接矩阵 (Distance) - 用于计算 Geodesic
    adj_dist = csr_matrix((data_dist, (row, col)), shape=(n_samples, n_samples))
    adj_dist = adj_dist + adj_dist.T
    
    # 4. 构建稀疏邻接矩阵 (Affinity) - 用于后处理平滑
    adj_aff = csr_matrix((data_aff, (row, col)), shape=(n_samples, n_samples))
    adj_aff = adj_aff + adj_aff.T
    
    # 5. 计算全源最短路径 (Geodesic Distance)
    # 这会算出“考虑到组织边界的物理穿梭距离”
    geo_matrix = shortest_path(adj_dist, directed=False)

    # 处理不连通图导致的 inf 值
    if np.isinf(geo_matrix).any():
        finite_vals = geo_matrix[np.isfinite(geo_matrix)]
        max_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
        geo_matrix[np.isinf(geo_matrix)] = max_val * 1.5

    return geo_matrix, adj_aff



def graph_smooth_results(features, adj_matrix, alpha=0.5, n_iter=2):
    """
    使用随机游走 (Random Walk) 平滑预测结果
    features: (n_spots, n_genes)
    adj_matrix: 稀疏邻接矩阵 (W)
    alpha: 传播概率 (0~1)，越大越平滑
    """
    from scipy.sparse import diags
    
    # 归一化邻接矩阵 P = D^-1 * A
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # 避免除零
    D_inv = diags(1.0 / degrees)
    P = D_inv.dot(adj_matrix)
    
    # 迭代平滑: F_{t+1} = (1-alpha)F_0 + alpha * P * F_t
    F = features.copy()
    F0 = features.copy()
    
    for _ in range(n_iter):
        F = (1 - alpha) * F0 + alpha * P.dot(F)
        
    return F
