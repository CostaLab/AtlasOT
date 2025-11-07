import scFUGW
import numpy as np
import scanpy as sc
import muon as mu
import pandas as pd
import anndata as ad





adata = mu.read_h5mu('../Code/10-BMMC/OP_Multiome.h5mu.gz')

# For atac peaks
atac_peaks = adata['atac'].copy()


# For atac gene activity
atac_gene = ad.AnnData(X = atac_peaks.obsm['gene_activities'].copy())
atac_gene.var.index = atac_peaks.uns['gene_activities_var_names'].copy()
atac_gene.obs = atac_peaks.obs.copy()

del atac_peaks.uns
del atac_peaks.obsm
del atac_peaks.var['gene_id']

# For RNA
rna = adata['rna']


print('读完数据了，开跑')




# In this dataset, the donor_id is called batch, the cell_type is celltype
# After preprocessing, here we use scFUGW.
# This function will split the data by donor_id, and will add "shareSpace" in atac_peaks.obsm
scFUGW.preprocess_data( rna=rna,
                        atac_peaks=atac_peaks,
                        atac_gene=atac_gene,
                        rna_cell_type = 'celltype',
                        top_genes = 5000,
                        sample_names = 'batch',
                        save_folder = 'data/RNA_ATAC-BMMC',
                        peaks_reduction = 'both',
                        random_seed = 3407,
                        save_counts = True)
print('The data preprocessing is done, the data is saved in data/RNA_ATAC-MI')


# Here run the Fused Unbalanced Gromov Wasserstien.
scFUGW.main_test(
    save_folder='data/RNA_ATAC-BMMC',
    rna_reduction='RNA_pca_l2_norm',
    atac_reduction='scopen',
    cell_type='celltype',
    alpha=0.9,
    rho=120,
    eps=1e-4,
    top_cells=20,
    mapping_direction='ATAC2RNA',
)

