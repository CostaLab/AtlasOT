import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors


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





def cosine_distance_tensor(X, Y):
    import torch.nn.functional as F

    # 转成 tensor 并确保 float32
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    # L2 归一化
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)

    # 计算余弦相似度矩阵
    sim_matrix = torch.mm(X_norm, Y_norm.T)

    # 余弦距离 = 1 - 余弦相似度
    dist_matrix = 1 - sim_matrix

    return dist_matrix  # torch.Tensor


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
