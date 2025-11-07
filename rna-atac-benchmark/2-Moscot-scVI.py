import warnings
import numpy as np
import anndata as ad
import scanpy as sc
import muon as mu
import os
import scvi
from moscot.problems.cross_modality import TranslationProblem

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


import random
import torch
random_seed = 3407
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)



dataset = 'BMMC'

if dataset == 'MI' or dataset == 'kidney1' or dataset == 'kidney2':
    cell_type = 'cell_type'
else:
    cell_type = 'celltype'

local_folder = '../data/RNA_ATAC-' + dataset
samples = os.listdir(local_folder)
samples.sort()



def get_acc(MP, MT, mappedData, truthData, top=1):
    length = len(MP)
    counts = 0
    
    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        if mappedData[i] == truthData[indexes].value_counts().index[0]:
            counts += 1
    return counts/length


acc_results = [[], []]
for sample in samples:
    mdata = mu.read_h5mu(f'{local_folder}/{sample}')


    atac_peaks = mdata['peaks'].copy()

    atac_gene = mdata['gene_activity'].copy()
    atac_gene.var.index = atac_gene.var.index.str.split(':', expand = True)
    atac_gene.var['features'] = atac_gene.var.index

    rna = mdata['gene_expression'].copy()
    rna.var.index = rna.var.index.str.split(':', expand = True)
    rna.var['features'] = rna.var.index



    #Align the rna and atac_gene
    # common_genes = list(set(rna.var.features).intersection(atac_gene.var.features))
    # atac_sub_adata = atac_gene[:,common_genes].copy()
    # adata_rna = rna[:,common_genes].copy()

    # # scVI
    cat_adata = ad.concat([atac_gene, rna], join="inner", label="batch") # atac att first, will be '0', RNA is '1'
    cat_adata.layers["counts"] = cat_adata.X.copy()
    sc.pp.normalize_total(cat_adata)
    sc.pp.log1p(cat_adata)

    scvi.model.SCVI.setup_anndata(cat_adata, layer="counts", batch_key="batch")
    model = scvi.model.SCVI(cat_adata)
    model.train(accelerator='cpu', plan_kwargs={"lr": 0.001})
    latent = model.get_latent_representation()
    atac_peaks.obsm['scVI_data'] = latent[cat_adata.obs['batch'] == '0'] # peaks
    rna.obsm[ 'scVI_data'] = latent[cat_adata.obs['batch'] == '1'] # RNA



    fgw_hyperparameters = {
        'epsilon' : 1e-6, # Control the spread of the transport, lower values correspond to tighter coupling
        'alpha' : 0.9 # Control the relative weight given to the Wasserstein term relative to the GW term. 0 (W only) to 1 (GW only)
    }

    # Initializing and solving the problem
    ftp = TranslationProblem(adata_src=atac_peaks, adata_tgt=rna)
    ftp = ftp.prepare(
        src_attr="scopen", tgt_attr="RNA_pca_l2_norm", joint_attr="scVI_data", cost = {"xy":"cosine", "x":"cosine", "y":"cosine"},
    )



    ftp = ftp.solve(**fgw_hyperparameters)



    # optimal transport matrix
    OTM = ftp[('src', 'tgt')].solution.transport_matrix

    weights=np.sum(OTM, axis = 1) # T is the optimal transport matrix.
    OTM = np.nan_to_num(OTM, nan=0, posinf=0, neginf=0)

    mapped_atac = np.dot(OTM, rna.obsm['RNA_pca_l2_norm']) / weights[:, None] # Y is the target data.

    atac_peaks.obsm['mapped_atac'] = np.array(mapped_atac)
    
    MP = atac_peaks.obsm['mapped_atac'].copy()
    MT = rna.obsm['RNA_pca_l2_norm'].copy()
    
    acc_results[0].append(len(MP))
    acc = get_acc(MP, MT, atac_peaks.obs[cell_type], rna.obs[cell_type], top=20)
    acc_results[1].append(acc)


print('The code is done!')
print('The result details here:\n', acc_results)
print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))
