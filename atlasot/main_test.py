import numpy as np
import pandas as pd
import mudata as mu
import warnings
import torch
import random
from fugw.mappings import FUGW
import os
from .evaluate import *
from .core import *

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)



def main_test(
        save_folder = 'RNA_ATAC_mudata',
        rna_reduction = 'RNA_pca_l2_norm',
        atac_reduction = 'scopen',
        cell_type='cell_type',
        alpha = 0.1,
        rho = 1.1,
        eps = 1e-5,
        top_cells = 20,
        mapping_direction = 'ATAC2RNA',
        random_seed = 3407,
        ):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    samples = os.listdir(save_folder)
    samples.sort()

    acc_results = [[], []] # first list for cell number of each sample, the second list for acc of each sample
    for sample in samples:
        mdata = mu.read_h5mu(f'{save_folder}/{sample}')

        atac_peaks = mdata['peaks'].copy()
        rna = mdata['gene_expression'].copy()

        mapped_data, pi = scFUGW(rna,
                                 atac_peaks,
                                 target_reduction=rna_reduction,
                                 source_reduction=atac_reduction,
                                 # mapping_direction=mapping_direction,
                                 # cell_type=cell_type,
                                 alpha=alpha,
                                 rho=rho,
                                 eps=eps)

        if mapping_direction == 'ATAC2RNA':
            atac_peaks.obsm['mapped_data'] = np.array(mapped_data)

            MP = atac_peaks.obsm['mapped_data']
            MT = rna.obsm[rna_reduction]

            acc_results[0].append(len(MP))
            acc = get_acc(MP, MT, atac_peaks.obs[cell_type], rna.obs[cell_type], top=top_cells)
            acc_results[1].append(acc)

            # if label_transfer == True:
            #     predicted_label = label_transfer(MP, MT, rna.obs[cell_type], top=top_cells)
            #     atac_peaks.obs["pred_cell_type"] = predicted_label


        elif mapping_direction == 'RNA2ATAC':
            rna.obsm['mapped_data'] = np.array(mapped_data)

            MP = rna.obsm['mapped_data']
            MT = atac_peaks.obsm[rna_reduction]

            acc_results[0].append(len(MP))
            acc = get_acc(MP, MT, rna.obs[cell_type], atac_peaks.obs[cell_type], top=top_cells)
            acc_results[1].append(acc)

        else:
            raise NotImplementedError("Please set mapping_direction to be 'ATAC2RNA' or 'RNA2ATAC'.")

    print('The code is done!')
    print('The result details here:\n', acc_results)
    print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
    print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))