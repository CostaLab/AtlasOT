import numpy as np
from muon import MuData
import scanpy as sc
import muon as mu
from sklearn import preprocessing as pp
import warnings
import os
from sklearn.decomposition import PCA
import scanpy.external as sce
import random
from .evaluate import *
from scopen.Main import scopen_dr
from scipy.sparse import issparse

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)





def preprocess_rna_atac(rna,
                    atac_peaks,
                    atac_gene,
                    rna_cell_type = 'cell_type',
                    top_genes = None,
                    sample_names = 'donor_id',
                    save_folder = 'RNA_ATAC_mudata',
                    peaks_reduction = 'both',
                    random_seed = 3407,
                    save_counts = True,
                    save_atac_gene = True):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if rna_cell_type not in rna.obs:
        raise ValueError(f"Not find the {rna_cell_type} in the rna.obs")
    # if atac_cell_type not in atac_peaks.obs:
    #     raise ValueError(f"Not find the {atac_cell_type} in the atac_peaks.obs")
    # if atac_cell_type not in atac_gene.obs:
    #     raise ValueError(f"Not find the {atac_cell_type} in the atac_gene.obs")

    if sample_names is not None:
        if sample_names not in rna.obs:
            raise ValueError(f"Not find the {sample_names} in the rna.obs")
        if sample_names not in atac_peaks.obs:
            raise ValueError(f"Not find the {sample_names} in the atac_peaks.obs")
        if sample_names not in atac_gene.obs:
            raise ValueError(f"Not find the {sample_names} in the atac_gene.obs")

        if not (set(rna.obs[sample_names].unique()) == set(atac_peaks.obs[sample_names].unique())
                and set(rna.obs[sample_names].unique()) == set(atac_gene.obs[sample_names].unique())):
            raise ValueError(f"The {sample_names} not match in rna.obs and atac.obs")

    # rna.X = rna.X.toarray() if issparse(rna.X) else np.asarray(rna.X)

    # Data preprocessing
    atac_peaks = preprocess_atac_peaks(atac_peaks, save_counts=save_counts)
    atac_peaks = reduction_atac_peaks(atac=atac_peaks, reduction=peaks_reduction,
                                      sample_id=sample_names, random_seed=random_seed)

    atac_gene = preprocess_atac_genes(atac_gene=atac_gene, save_counts=save_counts)
    atac_gene = reduction_atac_genes(atac_gene=atac_gene)

    rna = preprocess_rna(rna, save_counts=save_counts)
    rna = reduction_rna(rna, sample_id=sample_names, random_seed=random_seed)
    print('Preprocessing completed!')

    if top_genes is not None:
        sc.pp.highly_variable_genes(atac_gene, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=top_genes)
        common_genes = list(set(rna.var.index).intersection(atac_gene[:, atac_gene.var.highly_variable].var.index))

    else:
        # from common gene to build shared space
        common_genes = list(set(rna.var.features).intersection(atac_gene.var.features))
    print('The current length of common gene is: ', len(common_genes))
    if len(common_genes) == 0:
        raise ValueError('There are no common genes! Please make sure the gene name format is the same.')



    rna.obsm['shareSpace'], atac_peaks.obsm['shareSpace'] = find_shared_space(common_genes, rna, atac_gene, random_seed, control=None)
    print('Shared space established!')


    # save data
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if sample_names is not None:
        for sample_to_select in atac_peaks.obs[sample_names].unique():
            atac_peaks_sub = atac_peaks[atac_peaks.obs[sample_names] == sample_to_select].copy()
            atac_gene_activity_sub = atac_gene[atac_gene.obs[sample_names] == sample_to_select].copy()
            rna_sub = rna[rna.obs[sample_names] == sample_to_select].copy()

            if save_atac_gene:
                sample_mudata = MuData({"peaks": atac_peaks_sub, "gene_expression" : rna_sub, "gene_activity" : atac_gene_activity_sub})
            else:
                sample_mudata = MuData({"peaks": atac_peaks_sub, "gene_expression": rna_sub})

            sample_mudata.write(f'{save_folder}/atac_rna_{sample_to_select}.h5mu')
    else:
        if save_atac_gene:
            sample_mudata = MuData({"peaks": atac_peaks, "gene_expression": rna, "gene_activity" : atac_gene})
        else:
            sample_mudata = MuData({"peaks": atac_peaks, "gene_expression": rna})
        sample_mudata.write(f'{save_folder}/atac_rna_single.h5mu')





















def preprocess_atac_peaks(atac, save_counts = True):
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, "n_cells_by_counts", lambda x: x >= 10)
    if save_counts == True:
        atac.layers["counts"] = atac.X

    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)
    # sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=0.5)
    # atac.raw = atac
    return atac

def reduction_atac_peaks(atac, reduction = 'scopen', sample_id = None, lsi_comps = 101, random_seed = 3407):
    if reduction == 'scopen' or reduction == 'both':
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
def preprocess_atac_genes(atac_gene, save_counts = True):
    atac_gene.var.index = atac_gene.var.index.str.split(':', expand = True)
    atac_gene.var['features'] = atac_gene.var.index
    sc.pp.calculate_qc_metrics(atac_gene, percent_top=None, log1p=False, inplace=True)

    if save_counts == True:
        atac_gene.layers["counts"] = atac_gene.X

    sc.pp.normalize_per_cell(atac_gene, counts_per_cell_after=1e-4)
    sc.pp.log1p(atac_gene)
    return atac_gene

def reduction_atac_genes(atac_gene, sample_id = None, random_seed = 3407):
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
def preprocess_rna(rna, save_counts = True, filter=False):
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

def reduction_rna(rna, sample_id = None, random_seed = 3407):
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


def preprocess_spatial(spatial, save_counts = True):
    spatial.var.index = spatial.var.index.str.split(':', expand = True)
    spatial.var['features'] = spatial.var.index

    if save_counts == True:
        spatial.layers["counts"] = spatial.X

    sc.pp.normalize_total(spatial, inplace=True)
    sc.pp.log1p(spatial)
    return spatial

def reduction_spatial(spatial, sample_id = None, random_seed = 3407):

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


def find_shared_space(common_genes, rna, atac_gene, random_seed = 3407, control = None):
    sub_atac_gene = atac_gene[:, common_genes].copy()
    sc.pp.scale(sub_atac_gene)
    sub_rna = rna[:, common_genes].copy()
    sc.pp.scale(sub_rna)

    sub_atac_X = np.asarray(sub_atac_gene.X)
    sub_rna_X = np.asarray(sub_rna.X)
    if control is None:
        # 修复：确保PCA组件数不超过样本数和特征数的最小值
        print(f"Debug: len(common_genes)={len(common_genes)}, sub_rna_X.shape={sub_rna_X.shape}")
        max_components = min(len(common_genes), sub_rna_X.shape[0], sub_rna_X.shape[1], 500)
        print(f"Debug: max_components={max_components}")
        pca = PCA(n_components=max_components, random_state=random_seed)
        pca = pca.fit(np.asarray(sub_rna_X))
        rna_pca_explained_var = pca.explained_variance_ratio_
        index = find_threshold(rna_pca_explained_var)

        rna_pc = pca.transform(np.asarray(sub_rna_X))[:, :index]
        atac_pc = pca.transform(np.asarray(sub_atac_X))[:, :index]
    elif isinstance(control, int) and control > 0:
        pc = min(rna.X.shape[0], rna.X.shape[1], control)
        pca = PCA(n_components= pc, random_state=random_seed)
        pca = pca.fit(np.asarray(sub_rna_X))
        # rna_pca_explained_var = pca.explained_variance_ratio_
        # index = find_threshold(rna_pca_explained_var)

        rna_pc = pca.transform(np.asarray(sub_rna_X))#[:, :index]
        atac_pc = pca.transform(np.asarray(sub_atac_X))#[:, :index]
    else:
        raise ValueError(f"\"control\" should be None or positive integer, not {type(control)}")


    return rna_pc, atac_pc


