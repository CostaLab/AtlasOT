# AtlasOT — LLM Agent Reference (llms.txt compatible)

> Minimal context for AI coding agents (Claude, Copilot, etc.) to use this library.
> Human readers: see [README.md](./README.md)

## Purpose

AtlasOT aligns single-cell multi-modal data using Fused Unbalanced Gromov-Wasserstein (FUGW) optimal transport.
Two main tasks: **(A) RNA↔ATAC label transfer** and **(B) RNA→Spatial gene imputation**.

## Required `.obsm` / `.obsp` Keys

After preprocessing, AnnData objects must have these keys populated:

| Key | Location | Created by | Used by |
|-----|----------|------------|---------|
| `RNA_pca_l2_norm` | `rna.obsm` | `reduction_rna()` | `compute_geodesic_distance()`, label transfer |
| `scopen` | `atac.obsm` | `reduction_atac_peaks(reduction='scopen')` | `compute_geodesic_distance()` |
| `ATAC_pca_l2_norm` | `atac_gene.obsm` | `reduction_atac_genes()` | not directly by AtlasOT, but Harmony/batch correction |
| `shareSpace` | both `.obsm` | `find_shared_space()` | `cosine_distance_tensor()` → M matrix, `scFUGW_RNA_Spatial_with_cost()` |
| `spatial` | `sp.obsm` | pre-existing in data | `compute_spatial_geodesic()` |
| `cost_matrix` | both `.obsp` | `compute_geodesic_distance()` or `compute_spatial_geodesic()` | `scFUGW_RNA_Spatial_with_cost()` |

All cost matrices are **auto-normalized to [0, 1]** by `compute_geodesic_distance()` and `compute_spatial_geodesic()`.

## Task A: RNA → ATAC Label Transfer

```python
# 1. Load
m = mu.read_h5mu("sample.h5mu")
rna, atac, atac_gene = m['gene_expression'], m['peaks'], m['gene_activity']

# 2. Preprocess + reduce
rna       = aot.preprocess_rna(rna, save_counts=True)
rna       = aot.reduction_rna(rna)
atac      = aot.preprocess_atac_peaks(atac, save_counts=True)
atac      = aot.reduction_atac_peaks(atac, reduction='scopen')
atac_gene = aot.preprocess_atac_genes(atac_gene, save_counts=True)
atac_gene = aot.reduction_atac_genes(atac_gene)

# 3. Shared space
common = list(set(rna.var['features']) & set(atac_gene.var['features']))
rna.obsm['shareSpace'], atac.obsm['shareSpace'] = aot.find_shared_space(common, rna, atac_gene)

# 4. Cost matrices
M = aot.cosine_distance_tensor(atac.obsm['shareSpace'], rna.obsm['shareSpace'])
rna.obsp['cost_matrix'] = aot.compute_geodesic_distance(rna.obsm['RNA_pca_l2_norm'], k=30)
atac.obsp['cost_matrix'] = aot.compute_geodesic_distance(atac.obsm['scopen'], k=30)

# 5. Transport plan
pi = aot.scFUGW_RNA_Spatial_with_cost(target=rna, source=atac,
    target_cost='cost_matrix', source_cost='cost_matrix', M=M,
    alpha=0.1, rho=1.1, eps=1e-5).cpu().numpy()

# 6. Label transfer
pred = aot.ot_label_transfer(pi, rna.obs['cell_type'])
acc = (pred == atac.obs['cell_type'].values.astype(str)).mean()
```

**Direction**: `source=atac`, `target=rna` → π shape is `(n_atac, n_rna)`. Labels flow RNA→ATAC.

## Task B: RNA → Spatial Gene Imputation

```python
# 1. Load
m = mu.read_h5mu("sample.h5mu")
rna, sp = m['gene_expression'], m['spatial']

# 2. Preprocess + reduce (same functions work on spatial)
rna = aot.preprocess_rna(rna, save_counts=True)
rna = aot.reduction_rna(rna)
sp  = aot.preprocess_rna(sp, save_counts=True)
sp  = aot.reduction_rna(sp)

# 3. Shared space
common = list(set(rna.var['features']) & set(sp.var['features']))
rna.obsm['shareSpace'], sp.obsm['shareSpace'] = aot.find_shared_space(common, rna, sp)

# 4. Cost matrices
M = aot.cosine_distance_tensor(rna.obsm['shareSpace'], sp.obsm['shareSpace'])
rna.obsp['cost_matrix'] = aot.compute_geodesic_distance(rna.obsm['RNA_pca_l2_norm'], k=30)
sp.obsp['cost_matrix'], adj = aot.compute_spatial_geodesic(
    sp.obsm['spatial'], sp.obsm['RNA_pca_l2_norm'], k_phys=15)

# 5. Transport plan (RNA→Spatial: source=rna, target=sp)
pi = aot.scFUGW_RNA_Spatial_with_cost(target=sp, source=rna,
    target_cost='cost_matrix', source_cost='cost_matrix', M=M,
    alpha=0.5, rho=1.1, eps=1e-2).cpu().numpy()

# 6. Impute + smooth
imputed = aot.gene_imputation(pi, rna)
smoothed = aot.graph_smooth_results(imputed.values, adj, alpha=0.6, n_iter=2)
```

**Direction**: `source=rna`, `target=sp` → π shape is `(n_rna, n_spots)`. Genes flow RNA→Spatial via `gene_imputation`.

## Default Hyperparameters by Task

| Parameter | RNA-ATAC | RNA-Spatial |
|-----------|----------|-------------|
| `rho` | 1.1 | 1.1 |
| `eps` (start range) | 1e-6 ~ 1e-4 | 1e-2 ~ 1e-1 |
| `k` (geodesic) | 30 | 30 |
| `k_phys` (spatial) | — | 15 |
| `smooth_alpha` | — | 0.6 |
| `smooth_n_iter` | — | 2 |

### Tuning Guidance

**`eps` (entropic regularization)** — controls how sharp/blurry the transport plan is.
- **RNA-ATAC**: the two modalities are very different (peaks vs genes), so `eps` should be
  **smaller** to enforce stricter matching. Try **[1e-6, 1e-5, 1e-4]** first.
- **RNA-Spatial**: both are gene expression, modality gap is small, so `eps` can be larger.
  Try **[1e-2, 1e-1]** first.

**`alpha` (feature vs geometry weight)** — controls the trade-off between shared-space feature
matching (M) and intra-modality geometry preservation (C1, C2). α=0 means "trust M only",
α=1 means "trust C1/C2 only".

- **When cell differentiation is large / data is imbalanced**: the geometry graphs (C1, C2) become
  chaotic and unreliable — cells from very different lineages may appear geometrically disorder by chance. In this case, trust the shared-space features more: **α closer to 0**. Try values like
  0, 0.05, 0.1. 0 may not always be optimal — a small amount of geometry (e.g., α=0.05) can
  still help regularize.

- **When cell differentiation is small / data is multiome-like / balanced**: cells share similar
  expression patterns, making it hard to distinguish them from shared-space features alone. But
  their local geometric relationships (neighborhood structure) are stable and informative. In
  this case, trust geometry more: **α closer to 1**. Try values like 0.9, 0.95. Again, α=1
  may not be optimal — a small feature signal can help.

- **RNA-Spatial imputation**: start from **0.5** and tune in both directions — gene imputation
  needs a sweet spot that balances all four metrics (PCC, JS, RMSE, SSIM). Scan α ∈ [0.1, 0.9]
  to find the optimal trade-off for your data.

### Other Modality Pairs (not benchmarked but theoretically supported)

The same framework applies to any AnnData pair:
- **ATAC ↔ Spatial**: pick `eps` between the two ranges above since one side is peaks, the other genes.
- **RNA ↔ RNA** or **Spatial ↔ Spatial**: use the larger `eps` range (1e-2 ~ 1e-1), modalities match closely.

In general: **modality gap ↑ → eps ↓**, **shared-space quality ↑ → alpha ↑**.

## Public API Quick Reference

### Core (`atlasot.core`)
```python
scFUGW_RNA_Spatial_with_cost(target, source, target_cost='cost_matrix', source_cost='cost_matrix',
    alpha=0.6, rho=1.1, eps=1e-1, lambda_laplacian=5.0, random_seed=3407,
    M=None, L=None) -> torch.Tensor  # π (n_source × n_target)
```

### Cost Matrices (`atlasot.cost`)
```python
cosine_distance_tensor(X, Y) -> torch.Tensor              # cross-modality M
compute_geodesic_distance(X, k=30, metric='cosine') -> np.ndarray  # intra-modality C
compute_spatial_geodesic(coords, features, k_phys=10, metric='cosine') -> tuple[np.ndarray, csr_matrix]
compute_knn_normalized_laplacian(coords, n_neighbors=5, sigma=1.0) -> torch.Tensor
graph_smooth_results(features, adj_matrix, alpha=0.5, n_iter=2) -> np.ndarray
```

### Preprocessing (`atlasot.process_data`)
```python
preprocess_rna(rna, save_counts=True, filter=False) -> AnnData
reduction_rna(rna, sample_id=None, random_seed=3407) -> AnnData
preprocess_atac_peaks(atac, save_counts=True) -> AnnData
reduction_atac_peaks(atac, reduction='scopen', sample_id=None, lsi_comps=101, random_seed=3407) -> AnnData
preprocess_atac_genes(atac_gene, save_counts=True) -> AnnData
reduction_atac_genes(atac_gene, sample_id=None, random_seed=3407) -> AnnData
preprocess_spatial(spatial, save_counts=True) -> AnnData
reduction_spatial(spatial, sample_id=None, random_seed=3407) -> AnnData
find_shared_space(common_genes, rna, atac_gene, random_seed=3407, control=None) -> tuple[np.ndarray, np.ndarray]
```

### Downstream (`atlasot.evaluate`)
```python
ot_label_transfer(pi, target_labels) -> np.ndarray        # OT-based label transfer
gene_imputation(pi, source_adata) -> pd.DataFrame           # counts @ pi
map_to_target(pi, target_features) -> np.ndarray            # pi @ features (src→tgt; use pi.T for tgt→src)
label_transfer(MP, MT, truthData, top=1) -> list            # KNN-based label transfer
get_acc(MP, MT, mappedData, truthData, top=1) -> float      # KNN accuracy
find_threshold(values, threshold=1e-3) -> int | None        # PCA component selector

# Evaluation metrics (all return pd.DataFrame)
PCC(raw, impute)  JS(raw, impute, scale='scale_plus')
RMSE(raw, impute, scale='zscore')  SSIM(raw, impute, scale='scale_max')
compute_all_evaluation(raw, impute) -> None

# Normalization helpers
scale_plus(df)  scale_z_score(df)  scale_max(df)
cal_ssim(im1, im2, M) -> float
```

### Mini-Batch (`atlasot.minibatch`)
```python
split_indices(n_samples, n_batches, shuffle=True, random_seed=3407) -> list[np.ndarray]
compute_batch_cost_matrix(adata, modality='rna', k=30, device=None) -> np.ndarray
minibatch_atlasot(source_adata, target_adata, source_modality='rna', target_modality='rna',
    batch_source=1, batch_target=1, device=None, alpha=0.2, rho=1.1, eps=1e-2,
    random_seed=3407, verbose=True, projection_keys=None, label_columns=None, top_k=20
) -> tuple[pd.DataFrame|None, dict[str,np.ndarray], np.ndarray]
```

### Parameter Sweep (`atlasot.tune`)
```python
sweep_atlasot_alpha_eps(test_genes, rna, sp, alphas=None, epss=None,
    rho=1.1, lambda_laplacian=5.0, random_seed=3407, k_rna=30, k_phys=15,
    smooth_alpha=0.6, smooth_iter=2, spatial_key=None) -> pd.DataFrame  # (alpha, eps, PCC, JS, RMSE, SSIM)
```
End-to-end `alpha` x `eps` grid search for RNA->Spatial gene imputation. Feed it raw-count
`rna`/`sp` AnnData plus held-out `test_genes`; it runs the full pipeline internally
(preprocess -> shared space on train genes -> cost matrices -> impute -> graph smoothing),
prints the four metrics per combo, and returns the summary table.

### Plotting (`atlasot.plotting`)
```python
flip_visium_y(coords) -> np.ndarray        # Visium Y: origin top-left -> bottom-left
spatial_heatmap(coords, values, *, gene_names=None, title='', cmap='viridis',
    spot_size=10, save=None) -> plt.Figure # percentile-rank expression on tissue
spatial_deconvolution(pi, source_labels, coords, *, spot_size=10, save=None) -> (plt.Figure, pd.DataFrame)
dominant_type_map(pi, source_labels, coords, *, spot_size=12, save=None) -> (plt.Figure, pd.DataFrame)
```
Plot transport results on tissue: `pi` is (n_source, n_target), `source_labels` are the source
cell-type labels, `coords` are target-spot coordinates. All plots expect matplotlib convention
(Y up); call `flip_visium_y` first for Visium-style data.

## Constraints & Gotchas

1. **`find_shared_space` requires `var['features']`** — must call `preprocess_rna()` / `preprocess_atac_genes()` first (they set it).
2. **Cost matrices must be stored in `.obsp`** before calling `scFUGW_RNA_Spatial_with_cost`.
3. **Source/target direction matters**: π[i, j] is mass from source-cell-i to target-cell-j.
4. **`gene_imputation` uses `.X`** — by default it imputes from ``source_adata.X``, which after
   ``preprocess_rna()`` contains log-normalized expression. If you need to impute raw counts or
   any other matrix (e.g., a custom score matrix, TF activity, etc.), replace ``source_adata.X``
   with your matrix of choice before calling ``gene_imputation``. The function simply computes
   ``X.T @ pi``, so any (n_source × n_features) matrix works.
5. **`preprocess_rna` removes MT- genes** — if your data uses different mitochondrial gene naming, results may differ.
