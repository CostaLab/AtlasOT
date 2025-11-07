#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import chain

import anndata as ad
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
import scglue
import os
import seaborn as sns
from matplotlib import rcParams
import warnings
import numpy as np

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)



scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)


dataset = 'MI'


save_path = 'glueData-' + dataset

if dataset == 'MI' or dataset == 'kidney1' or dataset == 'kidney2':
    cell_type = 'cell_type'
else:
    cell_type = 'celltype'

local_folder = '../../data/RNA_ATAC-' + dataset
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


# In[ ]:

acc_results = [[], []]
for sample in samples:

    rna = ad.read_h5ad(f"{save_path}/rna-{sample[9:-5]}.h5ad")
    atac = ad.read_h5ad(f"{save_path}/atac-{sample[9:-5]}.h5ad")
    guidance = nx.read_graphml(f"{save_path}/guidance.graphml-{sample[9:-5]}.gz")

    scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="RNA_pca_l2_norm"
    )

    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="scopen"
    )

    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    fit_kws={"directory": "glue"}
    )

    dx = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)



    MP = atac.obsm["X_glue"]
    MT = rna.obsm["X_glue"]

    acc_results[0].append(len(MP))
    acc = get_acc(MP, MT, atac.obs[cell_type], rna.obs[cell_type], top=20)
    acc_results[1].append(acc)

print('The code is done!')
print('The result details here:\n', acc_results)
print('The weight accuracy is: ', np.average(acc_results[1], weights=acc_results[0]))
print('The mean accuracy is: ', sum(acc_results[1]) / len(acc_results[1]))
