# scFUGW
Graph4Patients Project, which is based on Optimal transport

# Structure
**Coin code**
- **scFUGW**: The coin code for the scFUGW tools
<br>

**RNA-ATAC**
- rna-atac-mapping: Map atac cells from original space to the target space
- rna-atac-benchmark: The benchmarking to compare with other RNA-ATAC integration algorithms
- rna-atac-analysis: Analyse the results and plot some figures.

<br>


**RNA-Spatial**
- rna-spatial-benchmark: Impute the spatial genes from RNA genes, to compare with other RNA-SPATIAL integration algorithms.

- rna-spatial-analysis: Analyse the results and plot some figures.


# Install
```python
conda create -n scfugw python=3.10

conda activate scfugw

pip install .
```
