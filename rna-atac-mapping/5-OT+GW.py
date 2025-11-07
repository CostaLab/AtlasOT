import scFUGW



scFUGW.main_test(
    save_folder='data/RNA_ATAC-MI',
    rna_reduction='RNA_pca_l2_norm',
    atac_reduction='scopen',
    cell_type='cell_type',
    alpha=0.0,
    rho=1.1,
    eps=1e-5,
    top_cells=20,
    mapping_direction='ATAC2RNA',
)
print('good')
