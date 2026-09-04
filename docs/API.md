# AtlasOT API Reference

> Version 0.3.0 — A library for single-cell multimodal integration via optimal transport.

## Core

### `scFUGW_RNA_Spatial_with_cost(target, source, target_cost='cost_matrix', source_cost='cost_matrix', alpha=0.6, rho=1.1, eps=1e-1, lambda_laplacian=5.0, random_seed=3407, M=None, L=None) → torch.Tensor`

Run Fused Unbalanced Gromov-Wasserstein alignment using pre-computed cost matrices.
Returns the transport plan π of shape (n_source × n_target).

`target` and `source` must have `obsm['shareSpace']` (shared latent space) and
`obsp[target_cost]` / `obsp[source_cost]` (intra-modality cost matrices).

---

## Cost Matrices

### `cosine_distance_tensor(X, Y) → torch.Tensor`

Pairwise cosine distance between two embedding matrices.
Returns shape (n_X, n_Y), values in [0, 2].

### `compute_geodesic_distance(X, k=30, metric='cosine') → np.ndarray`

KNN-graph shortest-path distance on embeddings. Auto-normalized to [0, 1].

### `compute_spatial_geodesic(coords, features, k_phys=10, metric='cosine') → (np.ndarray, csr_matrix)`

Physically-constrained geodesic distance for spatial data. Edges only exist between
physical neighbors; edge weights are expression-based. Returns `(geo_matrix, affinity_matrix)`.

### `compute_knn_normalized_laplacian(spatial_coords, n_neighbors=5, sigma=1.0) → torch.Tensor`

k-NN Gaussian-affinity normalized Laplacian from spatial coordinates.

### `graph_smooth_results(features, adj_matrix, alpha=0.5, n_iter=2) → np.ndarray`

Random-walk smoothing of imputed expression using spatial affinity matrix.

---

## Preprocessing

### `preprocess_rna(rna, save_counts=True, filter=False) → AnnData`

Normalize RNA: split gene names, remove MT- genes, normalize total, log1p.

### `reduction_rna(rna, sample_id=None, random_seed=3407) → AnnData`

PCA reduction + L2 normalization → `obsm['RNA_pca_l2_norm']`.
If `sample_id` is given, applies Harmony batch correction.

### `preprocess_atac_peaks(atac, save_counts=True) → AnnData`

Normalize ATAC peaks: filter low-coverage, normalize to 1e4 per cell, log1p.

### `reduction_atac_peaks(atac, reduction='scopen', sample_id=None, lsi_comps=101, random_seed=3407) → AnnData`

Reduce ATAC peaks via scOpen or LSI (or both).
Stores in `obsm['scopen']` and/or `obsm['ATAC_lsi_l2_norm']`.

### `preprocess_atac_genes(atac_gene, save_counts=True) → AnnData`

Normalize ATAC gene activity: split gene names, normalize to 1e4 per cell, log1p.

### `reduction_atac_genes(atac_gene, sample_id=None, random_seed=3407) → AnnData`

PCA reduction + L2 normalization → `obsm['ATAC_pca_l2_norm']`.

### `preprocess_spatial(spatial, save_counts=True) → AnnData`

Normalize spatial data: split gene names, normalize total, log1p.

### `reduction_spatial(spatial, sample_id=None, random_seed=3407) → AnnData`

PCA reduction + L2 normalization → `obsm['spatial_pca_l2_norm']`.

### `find_shared_space(common_genes, rna, atac_gene, random_seed=3407, control=None) → (np.ndarray, np.ndarray)`

Build shared latent space: subset to common genes, scale, fit PCA on RNA, project both.
Returns `(rna_pc, atac_pc)`.

---

## Label Transfer & Gene Imputation

### `ot_label_transfer(pi, target_labels) → np.ndarray`

Transfer labels from target to source via one-hot encoding + optimal transport.
Returns predicted labels for each source cell.

### `gene_imputation(pi, source_adata) → pd.DataFrame`

Impute gene expression: `source_adata.X.T @ pi`, returns DataFrame of shape (n_target, n_genes).

### `map_to_target(pi, target_features) → np.ndarray`

Map source cells to target feature space: `pi @ target_features / row_sum(pi)`.
Useful for projecting source embeddings (e.g., PCA, UMAP) into target coordinates.

### `label_transfer(MP, MT, truthData, top=1) → list`

KNN-based label transfer in embedding space.

### `get_acc(MP, MT, mappedData, truthData, top=1) → float`

KNN voting accuracy for label transfer evaluation. Returns value in [0, 1].

### `find_threshold(values, threshold=1e-3) → int | None`

Find the first index where a value drops below threshold. Used for PCA component selection.

---

## Evaluation Metrics

All metrics operate per-gene and return a `pd.DataFrame` (1 row × n_genes).

### `PCC(raw, impute) → pd.DataFrame`

Pearson correlation coefficient.

### `JS(raw, impute, scale='scale_plus') → pd.DataFrame`

Jensen-Shannon divergence. If `scale='scale_plus'`, normalizes columns to sum to 1 first.

### `RMSE(raw, impute, scale='zscore') → pd.DataFrame`

Root mean squared error. If `scale='zscore'`, z-score normalizes first.

### `SSIM(raw, impute, scale='scale_max') → pd.DataFrame`

Structural similarity index. If `scale='scale_max'`, normalizes to [0, 1] first.

### `compute_all_evaluation(raw, impute) → None`

Print all four metrics (PCC, JS, RMSE, SSIM) at once.

### `cal_ssim(im1, im2, M) → float`

Low-level SSIM computation between two 1D arrays.

---

## Normalization Helpers

### `scale_plus(df) → pd.DataFrame`

Normalize columns to sum to 1.

### `scale_z_score(df) → pd.DataFrame`

Z-score normalize columns.

### `scale_max(df) → pd.DataFrame`

Scale columns to [0, 1] by dividing by the maximum.

---

## Mini-Batch

### `split_indices(n_samples, n_batches, shuffle=True, random_seed=3407) → list[np.ndarray]`

Split sample indices into batch groups.

### `compute_batch_cost_matrix(adata, modality='rna', k=30, device=None) → np.ndarray`

Compute cost matrix for a data subset. Auto-selects embedding and method by modality
(`'rna'`, `'atac'`, or `'spatial'`).

### `minibatch_atlasot(source_adata, target_adata, source_modality='rna', target_modality='rna', batch_source=1, batch_target=1, device=None, alpha=0.2, rho=1.1, eps=1e-2, random_seed=3407, verbose=True, projection_keys=None, label_columns=None, top_k=20) → (...)`

Memory-efficient AtlasOT for large datasets via batch splitting.
Returns `(label_results_df, projected_results, full_target_features)`.

---

## Parameter Sweep

### `sweep_atlasot_alpha_eps(test_genes, rna, sp, alphas=None, epss=None, rho=1.1, lambda_laplacian=5.0, random_seed=3407, k_rna=30, k_phys=15, smooth_alpha=0.6, smooth_iter=2, spatial_key=None) → pd.DataFrame`

End-to-end `alpha` x `eps` grid search for RNA→Spatial gene imputation. Accepts
raw-count `rna` and `sp` AnnData plus held-out `test_genes`, and runs the full
pipeline: preprocess → shared space (trained on non-test genes) → cost matrices
→ AtlasOT → impute test genes → graph smoothing. Prints the four metrics per
combination and returns a summary table with columns `alpha, eps, PCC, JS,
RMSE, SSIM`.

`alpha` defaults to `0.0, 0.1, ..., 1.0`; `eps` defaults to `1e-1, 1e-2, 1e-3`.
`spatial_key` names the `.obsm` entry holding spot coordinates (default:
`'spatial'` if present, else `'X_spatial'`; raises `KeyError` if neither
exists).

---

## Plotting

Visualization of transport-plan results on spatial coordinates. All functions
expect coordinates in **matplotlib convention** (origin bottom-left, Y up); call
`flip_visium_y` first for Visium-style data (origin top-left, Y down).

### `flip_visium_y(coords) → np.ndarray`

Mirror the Y axis (`max(Y) - Y`) so Visium pixel coordinates render right-side
up in matplotlib.

### `spatial_heatmap(coords, values, *, gene_names=None, title='', cmap='viridis', spot_size=10, save=None) → plt.Figure`

Plot gene expression on tissue colored by percentile rank (robust to outliers,
comparable across genes). `values` may be `(n_spots,)` for one gene or
`(n_spots, n_genes)` for a facet grid.

### `spatial_deconvolution(pi, source_labels, coords, *, spot_size=10, save=None) → (plt.Figure, pd.DataFrame)`

Plot per-cell-type proportion on tissue: `onehot(source_labels).T @ pi`,
normalized so each spot's proportions sum to 1. Returns the figure and a
`(n_spots, n_types)` proportion DataFrame.

### `dominant_type_map(pi, source_labels, coords, *, spot_size=12, save=None) → (plt.Figure, pd.DataFrame)`

Color each spot by its dominant (highest transport mass) cell type, with a
per-type count legend. Returns the figure and the proportion DataFrame.
