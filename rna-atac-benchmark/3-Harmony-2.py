import scanpy.external as sce
import mudata as mu
import warnings
import scanpy as sc
import os
import numpy as np
import pandas as pd
from harmonypy import run_harmony

import random
random_seed = 3407
random.seed(random_seed)


def get_acc(MP, MT, mappedData, truthData, top=1):
    length = len(MP)
    counts = 0

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        if mappedData[i] == truthData[indexes].value_counts().index[0]:
            counts += 1
    return counts / length


dataset = 'MI'
save_path = 'glueData-' + dataset

if dataset == 'MI' or dataset == 'kidney1' or dataset == 'kidney2':
    cell_type = 'cell_type'
else:
    cell_type = 'celltype'

local_folder = '../data/RNA_ATAC-' + dataset
samples = os.listdir(local_folder)
samples.sort()


i=0

acc_results = [[], []]
for sample in samples:
    mdata = mu.read_h5mu(f'{local_folder}/{sample}')

    i += 1
    print('当前：', i)
    atac_gene = mdata['gene_activity']
    rna = mdata['gene_expression'].copy()

    l = len(rna)
    adata = sc.concat(
        [rna, atac_gene],
        join="outer",
        label="batch",
        keys=["RNA", "ATAC trans"],
        )

    adata.obsm["X_harmony"] = np.concatenate(
        (rna.obsm["RNA_pca_l2_norm"], atac_gene.obsm['ATAC_pca_l2_norm']), axis=0  # mapped data compare with the original data
    )

    sce.pp.harmony_integrate(adata, 'batch', "X_harmony", adjusted_basis='X_harmony')


    MP = adata.obsm['batch'][l:].to_numpy()
    MT = adata.obsm['batch'][:l].to_numpy()

    acc_results[0].append(len(MP))
    acc = get_acc(MP, MT, atac_gene.obs[cell_type], rna.obs[cell_type], top=20)
    acc_results[1].append(acc)

print('The code is done!')
print('The result details here:\n', acc_results)
print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))