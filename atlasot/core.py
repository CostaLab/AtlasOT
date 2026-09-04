from __future__ import annotations

import warnings
import torch
import random
from anndata import AnnData
from fugw.mappings import FUGW
from .evaluate import *


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)



def scFUGW_RNA_Spatial_with_cost(
        target: AnnData,
        source: AnnData,
        target_cost: str = 'cost_matrix',
        source_cost: str = 'cost_matrix',
        alpha: float = 0.6,
        rho: float = 1.1,
        eps: float = 1e-1,
        lambda_laplacian: float = 5.0,
        random_seed: int | None = 3407,
        M: torch.Tensor | None = None,
        L: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Fused Unbalanced Gromov-Wasserstein alignment with pre-computed cost matrices.

    Uses L2-normalized shared-space embeddings as feature representations and
    pre-computed geodesic cost matrices as intra-modality geometries.

    Parameters
    ----------
    target : AnnData
        Target modality (e.g., RNA or Spatial). Must have ``obsm['shareSpace']``
        and ``obsp[target_cost]``.
    source : AnnData
        Source modality (e.g., ATAC or RNA). Must have ``obsm['shareSpace']``
        and ``obsp[source_cost]``.
    target_cost : str
        Key in ``target.obsp`` for the target intra-modality cost matrix.
    source_cost : str
        Key in ``source.obsp`` for the source intra-modality cost matrix.
    alpha : float
        Weight of the geometry (Gromov-Wasserstein) term in the objective
        ``cost = (1 - alpha) * feature + alpha * geometry``. alpha = 0 uses only
        the cross-modality feature matrix ``M``; alpha = 1 uses only the
        intra-modality geometries (C1/C2).
    rho : float
        Unbalancedness parameter for marginal relaxation.
    eps : float
        Entropic regularization strength.
    lambda_laplacian : float
        Laplacian regularization weight (only used when ``L`` is provided).
    random_seed : int or None
        Random seed for reproducibility.
    M : torch.Tensor or None
        Pre-computed cross-modality feature distance matrix (source vs target).
        If None, computed internally.
    L : torch.Tensor or None
        Laplacian regularization matrix for the target modality.

    Returns
    -------
    torch.Tensor
        Optimal transport coupling matrix π (n_source × n_target).
    """

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)


    # target.var.index = target.var.index.str.split(':', expand=True)
    # target.var['features'] = target.var.index

    target_shareSpace = torch.from_numpy(target.obsm['shareSpace'])
    source_shareSpace = torch.from_numpy(source.obsm['shareSpace'])

    target_shareSpace = target_shareSpace / torch.linalg.norm(
        target_shareSpace, dim=1
    ).reshape(-1, 1)
    source_shareSpace = source_shareSpace / torch.linalg.norm(
        source_shareSpace, dim=1
    ).reshape(-1, 1)


    target_geometry = torch.tensor(target.obsp[target_cost], dtype=torch.float32)
    source_geometry = torch.tensor(source.obsp[source_cost], dtype=torch.float32)

    target_geometry = target_geometry / target_geometry.max()
    source_geometry = source_geometry / source_geometry.max()

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps, lambda_laplacian=lambda_laplacian)


    _ = mapping.fit(
        source_shareSpace.T,
        target_shareSpace.T,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        solver="sinkhorn",
        verbose=True,
        M = M,
        L = L,
    )

    pi = mapping.pi

    # mapped_data = np.dot(pi, target.obsm[target_reduction]) / pi.sum(dim=1).reshape(-1, 1)
    return pi
