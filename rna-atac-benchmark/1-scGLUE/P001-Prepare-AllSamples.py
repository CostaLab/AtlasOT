#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import numpy as np
import os
import mudata as mu
import scglue
import networkx as nx
import warnings

from muon import atac

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import random
import torch
random_seed = 3407
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)



dataset = 'MI'


local_folder = '../../data/RNA_ATAC-' + dataset
samples = os.listdir(local_folder)
samples.sort()

save_path = 'glueData-' + dataset


for sample in samples:
    mudata = mu.read_h5mu(f'{local_folder}/{sample}')
    
    atac_peaks = mudata['peaks'].copy()
    rna = mudata['gene_expression'].copy()
    
    if dataset == 'PBMC':
        del rna.var['strand']
        del rna.var['gene_ids']

    sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat_v3")
    scglue.data.get_gene_annotation(
    rna, gtf="gencode.v46lift37.annotation.gtf.gz",
    gtf_by="gene_name"
    )
    rna.var = rna.var.loc[:, ~rna.var.columns.duplicated()]
    rna = rna[:, rna.var['chrom'].notna()].copy()

    split = atac_peaks.var_names.str.split(r"[:-]")
    atac_peaks.var["chrom"] = split.map(lambda x: x[0])
    atac_peaks.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac_peaks.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
    atac_peaks.var.head()

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac_peaks)
    scglue.graph.check_graph(guidance, [rna, atac_peaks])

    if dataset == 'MI' or dataset == 'BMMC' or dataset == 'PBMC':
        del rna.var['artif_dupl']
    elif dataset == 'kidney1' or dataset == 'kidney2':
        del rna.var['artif_dupl']
        del rna.var['hgnc_id']
        del rna.var['gene_status']
        del rna.var['remap_substituted_missing_target']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rna.write(f"{save_path}/rna-{sample[9:-5]}.h5ad", compression="gzip")
    atac_peaks.write(f"{save_path}/atac-{sample[9:-5]}.h5ad", compression="gzip")
    nx.write_graphml(guidance, f"{save_path}/guidance.graphml-{sample[9:-5]}.gz")

