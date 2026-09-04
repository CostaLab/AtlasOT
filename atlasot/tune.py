"""Parameter sweep helpers for AtlasOT RNA -> Spatial gene imputation."""

import numpy as np
import pandas as pd

from .core import scFUGW_RNA_Spatial_with_cost
from .cost import (
    compute_geodesic_distance,
    compute_spatial_geodesic,
    graph_smooth_results,
    cosine_distance_tensor,
)
from .evaluate import PCC, JS, RMSE, SSIM, gene_imputation
from .process_data import (
    preprocess_rna,
    reduction_rna,
    preprocess_spatial,
    reduction_spatial,
    find_shared_space,
)


def sweep_atlasot_alpha_eps(
    test_genes,
    rna,
    sp,
    alphas=None,
    epss=None,
    rho=1.1,
    lambda_laplacian=5.0,
    random_seed=3407,
    k_rna=30,
    k_phys=15,
    smooth_alpha=0.6,
    smooth_iter=2,
    spatial_key=None,
):
    """Sweep ``alpha`` x ``eps`` for RNA->Spatial gene imputation.

    Runs the full AtlasOT pipeline on raw counts and reports the four imputation
    metrics (PCC, JS, RMSE, SSIM) for every (alpha, eps) combination.

    Parameters
    ----------
    test_genes : list of str
        Held-out genes, present in both ``rna`` and ``sp``. They are excluded
        from shared-space training but used for ground-truth evaluation.
        Names must match the (``:``-split) gene names, as in the existing scripts.
    rna : AnnData
        Raw-count RNA modality (cells x genes).
    sp : AnnData
        Raw-count spatial modality (spots x genes). Spot coordinates are read
        from ``sp.obsm[spatial_key]``.
    alphas : iterable, optional
        Values of ``alpha`` to try. Defaults to 0.0, 0.1, ..., 1.0.
    epss : iterable, optional
        Values of ``eps`` to try. Defaults to [1e-1, 1e-2, 1e-3].
    rho, lambda_laplacian, random_seed, k_rna, k_phys : float / int
        Fixed AtlasOT / graph parameters.
    smooth_alpha, smooth_iter : float / int
        Graph smoothing applied to the imputed result before evaluation.
    spatial_key : str, optional
        Key in ``sp.obsm`` that holds the spot coordinates. Different assays
        name it differently (Visium ``'spatial'``, Xenium/Slide-seq
        ``'X_spatial'``, ...). Defaults to ``'spatial'`` if present, else
        ``'X_spatial'``. Raise a ``KeyError`` if the resolved key is missing.

    Returns
    -------
    pd.DataFrame
        One row per (alpha, eps) combo with columns
        ``alpha, eps, PCC, JS, RMSE, SSIM``.
    """
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 1) if alphas is None else list(alphas)
    epss = [1e-1, 1e-2, 1e-3] if epss is None else list(epss)

    # 1. Preprocess + reduce (mutates copies only)
    rna = reduction_rna(preprocess_rna(rna.copy()), sample_id=None, random_seed=random_seed)
    sp = reduction_spatial(preprocess_spatial(sp.copy()), sample_id=None, random_seed=random_seed)

    # 2. Split genes: test genes are held out of shared-space training
    test_genes = [g for g in sp.var_names if g in set(test_genes)]
    if not test_genes:
        raise ValueError("None of test_genes found in sp.var_names after preprocessing.")
    train_genes = [g for g in sp.var_names if g not in set(test_genes)]
    common_genes = list(set(rna.var_names).intersection(train_genes))

    # 3. Shared space (trained on train genes only)
    rna.obsm['shareSpace'], sp.obsm['shareSpace'] = find_shared_space(
        common_genes, rna, sp[:, train_genes].copy(), random_seed=random_seed
    )

    # 4. Intra-modality costs + cross-modality distance M
    rna.obsp['cost_matrix'] = compute_geodesic_distance(rna.obsm['RNA_pca_l2_norm'], k=k_rna)
    # Resolve the spot-coordinates key. NB: never use `obsm.get(a, obsm[b])`
    # here - the default argument is evaluated eagerly, so a missing `b` would
    # raise even when `a` exists. Callers may pick the key (assays name it
    # differently); otherwise auto-detect the common ones.
    if spatial_key is None:
        spatial_key = 'spatial' if 'spatial' in sp.obsm else 'X_spatial'
    if spatial_key not in sp.obsm:
        raise KeyError(
            f"Spatial-coordinates key {spatial_key!r} not found in sp.obsm "
            f"(available: {sorted(sp.obsm)}). "
            f"Pass the correct one via `spatial_key=`."
        )
    coords = sp.obsm[spatial_key]
    sp.obsp['cost_matrix'], adj = compute_spatial_geodesic(
        coords, sp.obsm['spatial_pca_l2_norm'], k_phys=k_phys, metric='cosine'
    )
    M = cosine_distance_tensor(rna.obsm['shareSpace'], sp.obsm['shareSpace'])

    # 5. Ground truth for the held-out test genes
    real = sp[:, test_genes].to_df()

    # 6. Sweep alpha x eps
    rows = []
    for a in alphas:
        for e in epss:
            pi = scFUGW_RNA_Spatial_with_cost(
                sp, rna,
                target_cost='cost_matrix', source_cost='cost_matrix',
                M=M,
                alpha=float(a), rho=rho, eps=float(e),
                lambda_laplacian=lambda_laplacian, random_seed=random_seed,
            )
            pi = pi.cpu().numpy() if hasattr(pi, 'cpu') else np.asarray(pi)

            imputed = gene_imputation(pi, rna[:, rna.var_names.isin(test_genes)])
            imputed.index = sp.obs_names
            smoothed = graph_smooth_results(
                imputed.values, adj, alpha=smooth_alpha, n_iter=smooth_iter
            )
            imputed = pd.DataFrame(smoothed, index=imputed.index, columns=imputed.columns)

            spots = real.index.intersection(imputed.index)
            genes = real.columns.intersection(imputed.columns)
            real_a, imp_a = real.loc[spots, genes], imputed.loc[spots, genes]

            pcc = PCC(real_a, imp_a).mean().mean()
            js = JS(real_a, imp_a).mean().mean()
            rmse = RMSE(real_a, imp_a).mean().mean()
            ssim = SSIM(real_a, imp_a).mean().mean()

            print(f"alpha={a:.1f} eps={e:.0e}  PCC={pcc:.4f}  JS={js:.4f}  RMSE={rmse:.4f}  SSIM={ssim:.4f}")
            rows.append((a, e, pcc, js, rmse, ssim))

    return pd.DataFrame(rows, columns=['alpha', 'eps', 'PCC', 'JS', 'RMSE', 'SSIM'])
