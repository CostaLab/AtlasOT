import numpy as np
import mudata as mu
import warnings
import ot
import os

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import random
random_seed = 3407
np.random.seed(random_seed)
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



dataset = 'BMMC'
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

    atac_peaks = mdata['peaks'].copy()
    rna = mdata['gene_expression'].copy()

    # The cost matrix
    M = ot.dist(atac_peaks.obsm['shareSpace'], rna.obsm['shareSpace'], metric='sqeuclidean')
    C_rna = ot.dist(rna.obsm['RNA_pca_l2_norm'], metric='sqeuclidean')
    C_atac = ot.dist(atac_peaks.obsm['scopen'], metric='sqeuclidean')

    C_rna /= C_rna.max()
    C_atac /= C_atac.max()
    M /= M.max()

    OT, logw = ot.fused_gromov_wasserstein(M, C_atac, C_rna, loss_fun='square_loss', alpha=0.1, verbose=True, log=True)

    weights = np.sum(OT, axis=1)
    OT = np.nan_to_num(OT, nan=0, posinf=0, neginf=0)
    mapped_data = np.dot(OT, rna.obsm['RNA_pca_l2_norm']) / weights[:, None]

    atac_peaks.obsm['mapped_atac'] = np.array(mapped_data)
    MP = atac_peaks.obsm['mapped_atac']
    MT = rna.obsm['RNA_pca_l2_norm']

    acc_results[0].append(len(MP))
    acc = get_acc(MP, MT, atac_peaks.obs[cell_type], rna.obs[cell_type], top=20)
    acc_results[1].append(acc)

print('The code is done!')
print('The result details here:\n', acc_results)
print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))


