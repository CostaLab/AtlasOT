from __future__ import annotations

import numpy as np
from anndata import AnnData
import scanpy as sc
import muon as mu
from sklearn import preprocessing as pp
import warnings
from sklearn.decomposition import PCA
import scanpy.external as sce
import random
from .evaluate import *
from scipy.sparse import issparse

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)





def preprocess_atac_peaks(
    atac: AnnData,
    save_counts: bool = True,
) -> AnnData:
    """Normalize ATAC peak counts.

    Filters low-coverage peaks, normalizes to 1e4 counts per cell, and log1p transforms.

    Parameters
    ----------
    atac : AnnData
        Raw ATAC peak counts.
    save_counts : bool
        If True, store raw counts in ``atac.layers['counts']``.

    Returns
    -------
    AnnData
        Normalized ATAC data.
    """
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, "n_cells_by_counts", lambda x: x >= 10)
    if save_counts == True:
        atac.layers["counts"] = atac.X

    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)

    return atac

def reduction_atac_peaks(
    atac: AnnData,
    reduction: str = 'scopen',
    sample_id: str | None = None,
    lsi_comps: int = 101,
    random_seed: int = 3407,
) -> AnnData:
    """Reduce ATAC peaks to low-dimensional embeddings.

    Supports 'scopen', 'lsi', or 'both'. Optionally applies Harmony batch correction.

    Parameters
    ----------
    atac : AnnData
        Normalized ATAC data.
    reduction : str
        Method: 'scopen', 'lsi', or 'both'.
    sample_id : str or None
        Key in ``atac.obs`` for batch labels (triggers Harmony if not None).
    lsi_comps : int
        Number of LSI components.
    random_seed : int
        Random seed.

    Returns
    -------
    AnnData
        ATAC data with embeddings in ``obsm['scopen']`` and/or ``obsm['ATAC_lsi_l2_norm']``.
    """
    if reduction == 'scopen' or reduction == 'both':
        # scOpen is only needed for this reduction; import lazily so RNA/Spatial
        # users never require it to be installed.
        from scopen.Main import scopen_dr

        atac.obsm["scopen"] = pp.normalize(np.transpose(scopen_dr(np.transpose(atac.X), random_state = random_seed)), norm="l2")
        if sample_id is not None:
            sce.pp.harmony_integrate(atac, sample_id, "scopen", adjusted_basis = 'scopen')

    if reduction == 'lsi' or reduction == 'both':
        mu.atac.tl.lsi(atac, n_comps = lsi_comps)
        atac.obsm["ATAC_lsi_l2_norm"] = pp.normalize(atac.obsm['X_lsi'][:,1:], norm="l2")
        if sample_id is not None:
            sce.pp.harmony_integrate(atac, sample_id, "ATAC_lsi_l2_norm", adjusted_basis = 'ATAC_lsi_l2_norm')
    return atac

# Preprocessing ATAC Gene
def preprocess_atac_genes(
    atac_gene: AnnData,
    save_counts: bool = True,
) -> AnnData:
    """Normalize ATAC gene activity scores.

    Splits var index on ':' to extract gene names, normalizes to 1e4 per cell,
    and log1p transforms.

    Parameters
    ----------
    atac_gene : AnnData
        Raw ATAC gene activity matrix.
    save_counts : bool
        If True, store raw counts in ``atac_gene.layers['counts']``.

    Returns
    -------
    AnnData
        Normalized ATAC gene activity data.
    """
    atac_gene.var.index = atac_gene.var.index.str.split(':', expand = True)
    atac_gene.var['features'] = atac_gene.var.index
    sc.pp.calculate_qc_metrics(atac_gene, percent_top=None, log1p=False, inplace=True)

    if save_counts == True:
        atac_gene.layers["counts"] = atac_gene.X

    sc.pp.normalize_per_cell(atac_gene, counts_per_cell_after=1e4)
    sc.pp.log1p(atac_gene)
    return atac_gene

def reduction_atac_genes(
    atac_gene: AnnData,
    sample_id: str | None = None,
    random_seed: int = 3407,
) -> AnnData:
    """Reduce ATAC gene activity to PCA embeddings.

    Parameters
    ----------
    atac_gene : AnnData
        Normalized ATAC gene activity data.
    sample_id : str or None
        Key in ``atac_gene.obs`` for batch labels.
    random_seed : int
        Random seed.

    Returns
    -------
    AnnData
        ATAC gene activity with ``obsm['ATAC_pca_l2_norm']``.
    """
    pca_comps = min(atac_gene.X.shape[0], atac_gene.X.shape[1], 300)
    pca = PCA(n_components = pca_comps, random_state = random_seed)
    X = atac_gene.X.toarray() if issparse(atac_gene.X) else np.asarray(atac_gene.X)
    pca = pca.fit(X)
    atac_pca_explained_var = pca.explained_variance_ratio_
    index = find_threshold(atac_pca_explained_var)
    atac_gene.obsm["ATAC_pca_l2_norm"] = pca.transform(X)[:, :index]
    atac_gene.obsm["ATAC_pca_l2_norm"] = pp.normalize(
        atac_gene.obsm["ATAC_pca_l2_norm"], norm="l2"
    )

    if sample_id is not None:
        sce.pp.harmony_integrate(atac_gene, sample_id, "ATAC_pca_l2_norm", adjusted_basis = 'ATAC_pca_l2_norm')
    return atac_gene

# Preprocessing RNA
def preprocess_rna(
    rna: AnnData,
    save_counts: bool = True,
    filter: bool = False,
) -> AnnData:
    """Normalize RNA expression data.

    Splits var index on ':' to extract gene names, removes mitochondrial genes,
    normalizes total counts, and log1p transforms.

    Parameters
    ----------
    rna : AnnData
        Raw RNA counts.
    save_counts : bool
        If True, store raw counts in ``rna.layers['counts']``.
    filter : bool
        If True, apply basic cell/gene filtering.

    Returns
    -------
    AnnData
        Normalized RNA data.
    """
    rna.var.index = rna.var.index.str.split(':', expand = True)
    rna.var['features'] = rna.var.index
    if filter == True:
        sc.pp.filter_cells(rna, min_genes = 1)
        sc.pp.filter_genes(rna, min_counts = 10)
    non_mito_genes_list = [name for name in rna.var_names if not name.startswith('MT-')]
    rna = rna[:, non_mito_genes_list]

    if save_counts == True:
        rna.layers["counts"] = rna.X

    sc.pp.normalize_total(rna, inplace=True)
    sc.pp.log1p(rna)
    return rna

def reduction_rna(
    rna: AnnData,
    sample_id: str | None = None,
    random_seed: int = 3407,
) -> AnnData:
    """Reduce RNA expression to PCA embeddings with L2 normalization.

    Parameters
    ----------
    rna : AnnData
        Normalized RNA data.
    sample_id : str or None
        Key in ``rna.obs`` for batch labels.
    random_seed : int
        Random seed.

    Returns
    -------
    AnnData
        RNA data with ``obsm['RNA_pca_l2_norm']``.
    """
    pca_comps = min(rna.X.shape[0], rna.X.shape[1], 300)
    pca = PCA(n_components = pca_comps, random_state = random_seed)
    X = rna.X.toarray() if issparse(rna.X) else np.asarray(rna.X)
    pca = pca.fit(X)
    rna_pca_explained_var = pca.explained_variance_ratio_
    index = find_threshold(rna_pca_explained_var)
    rna.obsm["RNA_pca_l2_norm"] = pca.transform(X)[:, :index]
    rna.obsm["RNA_pca_l2_norm"] = pp.normalize(
        rna.obsm["RNA_pca_l2_norm"], norm="l2"
    )
    if sample_id is not None:
        sce.pp.harmony_integrate(rna, sample_id, "RNA_pca_l2_norm", adjusted_basis = 'RNA_pca_l2_norm')
    return rna


def preprocess_spatial(
    spatial: AnnData,
    save_counts: bool = True,
) -> AnnData:
    """Normalize spatial transcriptomics data.

    Splits var index on ':' to extract gene names, normalizes total counts,
    and log1p transforms.

    Parameters
    ----------
    spatial : AnnData
        Raw spatial expression data.
    save_counts : bool
        If True, store raw counts in ``spatial.layers['counts']``.

    Returns
    -------
    AnnData
        Normalized spatial data.
    """
    spatial.var.index = spatial.var.index.str.split(':', expand = True)
    spatial.var['features'] = spatial.var.index

    if save_counts == True:
        spatial.layers["counts"] = spatial.X

    sc.pp.normalize_total(spatial, inplace=True)
    sc.pp.log1p(spatial)
    return spatial

def reduction_spatial(
    spatial: AnnData,
    sample_id: str | None = None,
    random_seed: int = 3407,
) -> AnnData:
    """Reduce spatial expression to PCA embeddings with L2 normalization.

    Parameters
    ----------
    spatial : AnnData
        Normalized spatial data.
    sample_id : str or None
        Key in ``spatial.obs`` for batch labels.
    random_seed : int
        Random seed.

    Returns
    -------
    AnnData
        Spatial data with ``obsm['spatial_pca_l2_norm']``.
    """

    pca_comps = min(spatial.X.shape[0], spatial.X.shape[1], 300)
    pca = PCA(n_components = pca_comps, random_state = random_seed)
    X = spatial.X.toarray() if issparse(spatial.X) else np.asarray(spatial.X)
    pca = pca.fit(X)
    spatial_pca_explained_var = pca.explained_variance_ratio_
    index = find_threshold(spatial_pca_explained_var)
    spatial.obsm["spatial_pca_l2_norm"] = pca.transform(X)[:, :index]
    spatial.obsm["spatial_pca_l2_norm"] = pp.normalize(
        spatial.obsm["spatial_pca_l2_norm"], norm="l2"
    )
    if sample_id is not None:
        sce.pp.harmony_integrate(spatial, sample_id, "spatial_pca_l2_norm", adjusted_basis = 'spatial_pca_l2_norm')
    return spatial


def find_shared_space(
    common_genes: list[str],
    rna: AnnData,
    atac_gene: AnnData,
    random_seed: int = 3407,
    control: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a shared latent space from common genes via PCA.

    Subsets both modalities to common genes, scales, then fits PCA on RNA
    and projects both into the same PC space.

    Parameters
    ----------
    common_genes : list of str
        Gene names present in both modalities.
    rna : AnnData
        RNA (or reference) data.
    atac_gene : AnnData
        ATAC gene activity (or query) data.
    random_seed : int
        Random seed for PCA.
    control : int or None
        If int, use exactly that many PCs. If None, auto-select via variance threshold.

    Returns
    -------
    rna_pc : np.ndarray
        Shared-space coordinates for RNA (n_rna, n_pcs).
    atac_pc : np.ndarray
        Shared-space coordinates for ATAC/spatial (n_atac, n_pcs).
    """
    sub_atac_gene = atac_gene[:, common_genes].copy()
    sc.pp.scale(sub_atac_gene)
    sub_rna = rna[:, common_genes].copy()
    sc.pp.scale(sub_rna)

    sub_atac_X = np.asarray(sub_atac_gene.X)
    sub_rna_X = np.asarray(sub_rna.X)
    if control is None:
        # Cap components at min(samples, features, 500)
        max_components = min(len(common_genes), sub_rna_X.shape[0], sub_rna_X.shape[1], 500)
        pca = PCA(n_components=max_components, random_state=random_seed)
        pca = pca.fit(np.asarray(sub_rna_X))
        rna_pca_explained_var = pca.explained_variance_ratio_
        index = find_threshold(rna_pca_explained_var)

        rna_pc = pca.transform(np.asarray(sub_rna_X))[:, :index]
        atac_pc = pca.transform(np.asarray(sub_atac_X))[:, :index]
    elif isinstance(control, int) and control > 0:
        pc = min(sub_rna_X.shape[0], sub_rna_X.shape[1], control)
        pca = PCA(n_components= pc, random_state=random_seed)
        pca = pca.fit(np.asarray(sub_rna_X))

        rna_pc = pca.transform(np.asarray(sub_rna_X))#[:, :index]
        atac_pc = pca.transform(np.asarray(sub_atac_X))#[:, :index]
    else:
        raise ValueError(f"\"control\" should be None or positive integer, not {type(control)}")


    return rna_pc, atac_pc


