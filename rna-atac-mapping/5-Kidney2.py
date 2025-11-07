import scFUGW
import numpy as np
import scanpy as sc
import muon as mu
import pandas as pd
import anndata as ad


atac_peaks = ad.read_h5ad("../Code/17-diabetic/data/atac_peaks_final.h5ad")
atac_gene = ad.read_h5ad("../Code/17-diabetic/data/atac_gene_final.h5ad")
rna = ad.read_h5ad('../Code/17-diabetic/data/rna_final.h5ad')



atac_peaks.obs['donor_id'] = atac_gene.obs['donor_id']

rna.var['features'] = rna.var.index
donor_ids = atac_gene.obs['donor_id'].unique()
rna = rna[rna.obs['donor_id'].isin(donor_ids)].copy()




print('读完数据了，开跑')


# In this dataset, the donor_id is called donor_uuid
# After preprocessing, here we use scFUGW.
# This function will split the data by donor_id, and will add "shareSpace" in atac_peaks.obsm
scFUGW.preprocess_data( rna=rna,
                        atac_peaks=atac_peaks,
                        atac_gene=atac_gene,
                        rna_cell_type = 'cell_type',
#                        atac_cell_type = 'cell_type',
                        sample_names = 'donor_id',
                        save_folder = 'data/RNA_ATAC-Kidney2',
                        peaks_reduction = 'both',
                        random_seed = 3407,
                        save_counts = True)
print('The data preprocessing is done, the data is saved in data/RNA_ATAC-Kidney1')


# Here run the Fused Unbalanced Gromov Wasserstien.
scFUGW.main_test(
    save_folder='data/RNA_ATAC-Kidney2',
    rna_reduction='RNA_pca_l2_norm',
    atac_reduction='scopen',
    cell_type='cell_type',
    alpha=0.1,
    rho=1.1,
    eps=1e-6,
    top_cells=20,
    mapping_direction='ATAC2RNA',
)



