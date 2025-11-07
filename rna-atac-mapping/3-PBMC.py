import scFUGW
import numpy as np
import scanpy as sc
import muon as mu
import pandas as pd
import anndata as ad




adata = mu.read_h5mu('../Code/11-PBMC/pbmc10X.h5mu')

atac_peaks = adata['atac'].copy()

atac_gene = ad.AnnData(X = adata['atac'].obsm['gene_activities'].copy().astype(float))
atac_gene.var.index = adata['atac'].uns['gene_activities_var_names'].copy()
atac_gene.obs = adata['atac'].obs.copy()

del atac_peaks.uns
del atac_peaks.obsm
del atac_peaks.var['gene_ids']

rna = adata['rna']



print('读完数据了，开跑')


# This dataset only one sample, the cell_type is celltype
# After preprocessing, here we use scFUGW.
# This function will split the data by donor_id, and will add "shareSpace" in atac_peaks.obsm
scFUGW.preprocess_data( rna=rna,
                        atac_peaks=atac_peaks,
                        atac_gene=atac_gene,
                        rna_cell_type = 'celltype',
                        top_genes = 5000,
                        sample_names = None,
                        save_folder = 'data/RNA_ATAC-PBMC',
                        peaks_reduction = 'both',
                        random_seed = 3407,
                        save_counts = True)
print('The data preprocessing is done, the data is saved in data/RNA_ATAC-MI')


# Here run the Fused Unbalanced Gromov Wasserstien.
scFUGW.main_test(
    save_folder='data/RNA_ATAC-PBMC',
    rna_reduction='RNA_pca_l2_norm',
    atac_reduction='scopen',
    cell_type='celltype',
    alpha=0.9,
    rho=120,
    eps=1e-4,
    top_cells=20,
    mapping_direction='ATAC2RNA',
)