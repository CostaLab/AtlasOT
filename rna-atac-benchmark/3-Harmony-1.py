import mudata as mu
import warnings
import os
import numpy as np
import pandas as pd
from harmonypy import run_harmony

import random
random_seed = 3407
random.seed(random_seed)

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def get_acc(MP, MT, mappedData, truthData, top=1):
    length = len(MP)
    counts = 0

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        if mappedData[i] == truthData[indexes].value_counts().index[0]:
            counts += 1
    return counts / length

def format_pca_df(pc_matrix, cell_series, modality):
    l_df = pd.DataFrame(pc_matrix, columns = [f'PC{x}' for x in range(1, pc_matrix.shape[1]+1)])
    l_df = pd.concat([l_df, cell_series.reset_index(drop = True)], axis = 'columns', ignore_index = False)
    l_df['modality'] = modality
    return(l_df)


dataset = 'MI'
save_path = 'glueData-' + dataset

if dataset == 'MI' or dataset == 'kidney1' or dataset == 'kidney2':
    cell_type = 'cell_type'
else:
    cell_type = 'celltype'

local_folder = '../data/RNA_ATAC-' + dataset
samples = os.listdir(local_folder)
samples.sort()



acc_results = [[], []]
for sample in samples:
    mdata = mu.read_h5mu(f'{local_folder}/{sample}')

    # atac_peaks = mdata['peaks'].copy()
    atac_gene = mdata['gene_activity']
    rna = mdata['gene_expression'].copy()

    l = len(rna)
    min_length = min(rna.obsm['RNA_pca_l2_norm'].shape[1], atac_gene.obsm['ATAC_pca_l2_norm'].shape[1])
    rna_pc_df = format_pca_df(pc_matrix=rna.obsm['RNA_pca_l2_norm'][:, :min_length],
                              cell_series=rna.obs.cell_type.astype(str),
                              modality='rna'
                              )
    atac_pc_df = format_pca_df(pc_matrix=atac_gene.obsm['ATAC_pca_l2_norm'][:, :min_length],
                               cell_series=atac_gene.obs.cell_type.astype(str),
                               modality='atac'
                               )

    plotting_data = pd.concat([rna_pc_df, atac_pc_df], axis='rows', ignore_index=False)
    print(plotting_data)
    data_mat = np.array(plotting_data.iloc[:, :-2])
    ho = run_harmony(data_mat, plotting_data, ['modality'], nclust=7, max_iter_kmeans=30, max_iter_harmony=20)
    res = pd.DataFrame(ho.Z_corr).T

    MP = res[l:].to_numpy()
    MT = res[:l].to_numpy()

    acc_results[0].append(len(MP))
    acc = get_acc(MP, MT, atac_gene.obs[cell_type], rna.obs[cell_type], top=20)
    acc_results[1].append(acc)

print('The code is done!')
print('The result details here:\n', acc_results)
print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))
