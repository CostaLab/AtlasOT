import scFUGW
import numpy as np
import scanpy as sc
import muon as mu
import pandas as pd
import anndata as ad

metadata = pd.read_csv('../bcaron/data/metadata/multimodal_sample_match_complete.tsv', sep = '\t', header = 0)
cell_type_to_select = ['CM', 'Myeloid', 'Endo', 'Fib', 'Pericyte', 'Neuronal', 'Lymphoid', 'vSMCs']
sample_to_select = ['FZ_GT_P4', 'FZ_GT_P19', 'FZ_P14', 'FZ_P18', 'FZ_P20', 'GT_IZ_P9',
       'GT_IZ_P9_rep2', 'GT_IZ_P13', 'GT_IZ_P15', 'IZ_BZ_P2', 'IZ_P3',
       'IZ_P10', 'IZ_P15', 'IZ_P16', 'RZ_BZ_P2', 'RZ_BZ_P3', 'RZ_BZ_P12',
       'RZ_FZ_P5', 'RZ_GT_P2', 'RZ_P3', 'RZ_P6', 'RZ_P9', 'RZ_P11',
       'control_P1', 'control_P7', 'control_P8', 'control_P17']


# Load atac peaks
atac = ad.read_h5ad('../bcaron/data/snATAC/atac_annotated_corrected_ids.h5ad')
atac.__dict__['_raw'].__dict__['_var'] = atac.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
atac = atac[atac.obs.cell_type.isin(cell_type_to_select)]
atac = atac[atac.obs['patient_region_id'].isin(sample_to_select)].copy()

# Load scATAC gene activity
atac_gene_activity = ad.read_h5ad('../bcaron/data/snATAC/atac_gene_activity.h5ad')
atac_gene_activity.__dict__['_raw'].__dict__['_var'] = atac_gene_activity.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

# Add correct donor_id
atac_gene_activity.obs.rename(columns={"orig.ident": "orig_ident"}, inplace=True)
atac_gene_activity.obs['patient_region_id'] = atac_gene_activity.obs.Sample
atac_gene_activity.obs = atac_gene_activity.obs.replace({'patient_region_id' : dict(zip(metadata.atac_id, metadata.patient_region_id))})
atac_gene_activity = atac_gene_activity[atac_gene_activity.obs['patient_region_id'].isin(sample_to_select)].copy()


# align the atac cells
common_cells = atac.obs_names.intersection(atac_gene_activity.obs_names)
atac = atac[common_cells, :]
atac_gene_activity = atac_gene_activity[common_cells, :]
atac_gene_activity.obs['cell_type'] = atac.obs.loc[atac_gene_activity.obs.index, 'cell_type']


atac.obs['cell_type'] = atac.obs['cell_type'].astype('category')
atac.obs['patient_region_id'] = atac.obs['patient_region_id'].astype('category')
atac_gene_activity.obs['cell_type'] = atac_gene_activity.obs['cell_type'].astype('category')
atac_gene_activity.obs['patient_region_id'] = atac_gene_activity.obs['patient_region_id'].astype('category')



# Load RNA
rna = ad.read_h5ad('../bcaron/data/snRNA/snRNAseq_cellsWithSubtype.h5ad')
rna.var['features'] = rna.var.index
rna.obs['cell_type'] = rna.obs.cell_type.str.replace('PC', 'Pericyte')
del rna.obsm['HARMONY']
del rna.obsm['PCA']
del rna.obsm['UMAP_HARMONY']

rna.obs.rename(columns={'patient_region_id': 'donor_id'}, inplace=True)
atac.obs.rename(columns={'patient_region_id': 'donor_id'}, inplace=True)
atac_gene_activity.obs.rename(columns={'patient_region_id': 'donor_id'}, inplace=True)







# 仅挑选几个做测试
selected_donors = ['control_P1']

# 仅保留指定 donor_id 的细胞
rna = rna[rna.obs['donor_id'].isin(selected_donors)].copy()
atac = atac[atac.obs['donor_id'].isin(selected_donors)].copy()
atac_gene_activity = atac_gene_activity[atac_gene_activity.obs['donor_id'].isin(selected_donors)].copy()








# After preprocessing, here we use scFUGW.


# This function will split the data by donor_id, and will add "shareSpace" in atac.obsm
scFUGW.preprocess_data( rna=rna,
                        atac_peaks=atac,
                        atac_gene=atac_gene_activity,
                        rna_cell_type = 'cell_type',
                        # atac_cell_type = 'cell_type',
                        sample_names = 'donor_id',
                        save_folder = 'data/RNA_ATAC-MI-Test',
                        peaks_reduction = 'both',
                        random_seed = 3407,
                        save_counts = True)
print('The data preprocessing is done, the data is saved in data/RNA_ATAC-MI')


# Here run the Fused Unbalanced Gromov Wasserstien.
scFUGW.main_test(
    save_folder='data/RNA_ATAC-MI-Test',
    rna_reduction='RNA_pca_l2_norm',
    atac_reduction='scopen',
    cell_type='cell_type',
    alpha=0.1,
    rho=1.1,
    eps=1e-5,
    top_cells=20,
    mapping_direction='ATAC2RNA',
)



