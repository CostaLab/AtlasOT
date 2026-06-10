import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, issparse
from .cost import compute_geodesic_distance, compute_spatial_geodesic, cosine_distance_tensor
from .process_data import find_shared_space
from .core import scFUGW_RNA_Spatial_with_cost
from .evaluate import label_transfer


def split_indices(n_samples, n_batches, shuffle=True, random_seed=3407):
    """
    Split indices into n_batches.
    """
    np.random.seed(random_seed)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    if n_batches <= 1:
        return [indices]
    
    return np.array_split(indices, n_batches)


def compute_batch_cost_matrix(adata, modality='rna', k=30, device=None):
    """
    Compute cost matrix for a batch based on modality.
    
    Args:
        adata: AnnData object for the batch.
        modality: 'rna', 'atac', or 'spatial'.
        k: Number of neighbors for graph construction.
    
    Returns:
        cost_matrix: Computed cost matrix (numpy array).
    """
    # 1. Determine Embedding Key
    emb_key = None
    if modality == 'rna':
        candidates = ['RNA_pca_l2_norm', 'X_pca', 'X_scvi']
    elif modality == 'atac':
        candidates = ['scopen', 'ATAC_pca_l2_norm', 'X_lsi', 'X_pca']
    elif modality == 'spatial':
        candidates = ['RNA_pca_l2_norm', 'X_pca'] # Spatial often uses expression PCA for feature similarity part
    else:
        candidates = ['X_pca']

    for key in candidates:
        if key in adata.obsm:
            emb_key = key
            break
            
    if emb_key is None:
        # Fallback to X if dense and small enough, or error?
        # Assuming X is features.
        features = adata.X
        if issparse(features):
            features = features.toarray()
    else:
        features = adata.obsm[emb_key]

    # 2. Compute Cost Matrix based on Modality
    if modality in ['rna', 'atac']:
        # Standard Geodesic Distance on kNN graph of features
        cost_matrix = compute_geodesic_distance(features, k=k)
        
    elif modality == 'spatial':
        # Spatial Geodesic: Mix of Physical and Feature distance
        # Requires 'spatial' or 'X_spatial' in obsm
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
        elif 'X_spatial' in adata.obsm:
            spatial_coords = adata.obsm['X_spatial']
        else:
            raise ValueError("Modality is 'spatial' but no spatial coordinates found in .obsm['spatial']")
            
        cost_matrix, _ = compute_spatial_geodesic(spatial_coords, features, k_phys=15, metric='cosine')
        
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Normalize
    if cost_matrix.max() > 0:
        cost_matrix /= cost_matrix.max()
        
    return cost_matrix.astype(np.float32)


def minibatch_scfugw(source_adata, target_adata, 
                     source_modality='rna', target_modality='rna',
                     batch_source=1, batch_target=1, 
                     device=None, 
                     alpha=0.2, rho=1.1, eps=1e-2, 
                     random_seed=3407,
                     verbose=True,
                     projection_keys=None,
                     label_columns=None,
                     top_k=20):
    """
    Run scFUGW in minibatches with modality-aware cost computation.
    
    Args:
        source_adata: Source AnnData.
        target_adata: Target AnnData.
        source_modality: 'rna' or 'atac'.
        target_modality: 'rna', 'atac', or 'spatial'.
        batch_source: Number of batches for source.
        batch_target: Number of batches for target.
        device: torch device.
        alpha, rho, eps: scFUGW parameters.
        projection_keys: List of keys in target_adata to project. 
                         e.g. ['X_pca', 'X']. 
                         If None, defaults to projecting the features used for alignment (e.g. 'RNA_pca_l2_norm').
        label_columns: List of label column names in target_adata.obs to transfer.
                       e.g. ['cluster_anno_l2', 'cluster_anno_coarse', 'cluster_anno_l1'].
                       If None, label transfer is skipped.
        top_k: Number of nearest neighbors for label transfer voting (default 20).
        
    Returns:
        label_results_df: DataFrame with label transfer results (spot_id + label columns).
                          None if label_columns is None.
        projected_results: Dictionary {key: projected_data_array}.
        full_target_features: Target features used for alignment.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Running Minibatch scFUGW ({source_modality} -> {target_modality})")
        print(f"Batches: Source={batch_source}, Target={batch_target}")
        print(f"Device: {device}")

    # 1. Prepare Indices
    source_indices_list = split_indices(source_adata.shape[0], batch_source, shuffle=True, random_seed=random_seed)
    target_indices_list = split_indices(target_adata.shape[0], batch_target, shuffle=True, random_seed=random_seed)
    
    batches_per_target = int(np.ceil(batch_source / batch_target))
    
    # 2. Identify Target Features for Alignment
    if target_modality == 'spatial':
        tgt_feat_key = 'RNA_pca_l2_norm' if 'RNA_pca_l2_norm' in target_adata.obsm else 'X_pca'
    elif target_modality == 'rna':
        tgt_feat_key = 'RNA_pca_l2_norm' if 'RNA_pca_l2_norm' in target_adata.obsm else 'X_pca'
    elif target_modality == 'atac':
        tgt_feat_key = 'scopen' if 'scopen' in target_adata.obsm else 'ATAC_pca_l2_norm'
    else:
        tgt_feat_key = 'X_pca'
        
    # 3. Determine Projection Keys
    if projection_keys is None:
        projection_keys = [tgt_feat_key]
    
    # Initialize storage for results
    projected_results = {}
    for key in projection_keys:
        # Determine shape
        if key in target_adata.obsm:
            n_feats = target_adata.obsm[key].shape[1]
        elif key == 'X':
            n_feats = target_adata.shape[1]
        else:
            # Try obs? Not supported yet for simplicity
            print(f"Warning: Key {key} not found in obsm or X. Skipping.")
            continue
            
        projected_results[key] = np.zeros((source_adata.shape[0], n_feats), dtype=np.float32)

    # 4. Initialize label transfer storage
    all_batch_label_results = []

    # 5. Iterate Batches
    # Logic: 
    # Since batches are randomly shuffled and distributions are assumed similar,
    # we just need to ensure:
    # 1. Every Source Batch gets matched to SOME Target Batch(es).
    # 2. Every Target Batch is used at least once (to cover full target space).
    # 3. Distribution is as even as possible.
    
    # We use a simple modulo-based round-robin assignment.
    # Source Batch i matches Target Batch (i % batch_target).
    # But if batch_target > batch_source, we need to assign multiple target batches to one source batch
    # to ensure all target batches are used.
    
    # Strategy:
    # We iterate through all Target Batches (0..batch_target-1) and assign them to Source Batches.
    # Since scFUGW loop is driven by Source Batches, we need to pre-calculate the mapping:
    # Source_Batch_Index -> List of Target_Batch_Indices
    
    s_to_t_map = {i: [] for i in range(batch_source)}
    
    # Distribute Target Batches to Source Batches evenly
    for t_idx in range(batch_target):
        # Assign target batch t_idx to source batch (t_idx % batch_source)
        # This ensures even load balancing and full coverage of target batches.
        s_idx = t_idx % batch_source
        s_to_t_map[s_idx].append(t_idx)
        
    # Now iterate source batches and use the assigned target batches
    for i, s_idxs in enumerate(source_indices_list):
        target_batch_indices = s_to_t_map[i]
        
        # If for some reason a source batch has no target batches assigned 
        # (e.g. batch_source > batch_target), we must assign at least one.
        # Fallback to round-robin: i % batch_target
        if not target_batch_indices:
            target_batch_indices = [i % batch_target]
            
        # Collect all cells from the assigned target batches
        t_idxs = []
        for t_b_idx in target_batch_indices:
            t_idxs.extend(target_indices_list[t_b_idx])
            
        if verbose:
            t_str = ",".join(map(str, [x+1 for x in target_batch_indices]))
            print(f"  Batch {i+1}/{batch_source} (n={len(s_idxs)}) -> Target Batches [{t_str}] (n={len(t_idxs)})")
            
        # Subset
        source_sub = source_adata[s_idxs].copy()
        target_sub = target_adata[t_idxs].copy()
        
        # Clean NaN values if any
        def clean_nan_values(adata):
            """Remove NaN and Inf values from X matrix"""
            from scipy.sparse import csr_matrix, issparse
            if issparse(adata.X):
                # Convert to dense, clean, then convert back to sparse
                X_dense = adata.X.toarray()
                X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
                adata.X = csr_matrix(X_dense)
            else:
                if hasattr(adata.X, 'toarray'):
                    adata.X = adata.X.toarray()
                adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            return adata
        
        source_sub = clean_nan_values(source_sub)
        target_sub = clean_nan_values(target_sub)
        
        # Compute Cost Matrices
        source_cost = compute_batch_cost_matrix(source_sub, source_modality, k=30)
        target_cost = compute_batch_cost_matrix(target_sub, target_modality, k=30)
        
        source_sub.obsp['cost_matrix'] = source_cost
        target_sub.obsp['cost_matrix'] = target_cost
        
        # Compute Shared Space (Local)
        # Check if shareSpace exists, if not compute it locally
        if 'shareSpace' not in source_sub.obsm or 'shareSpace' not in target_sub.obsm:
             if verbose: print("    Computing local shareSpace...")
             common_genes = list(set(source_sub.var_names) & set(target_sub.var_names))
             if len(common_genes) > 10:
                 try:
                     source_sub.obsm['shareSpace'], target_sub.obsm['shareSpace'] = find_shared_space(
                        common_genes, source_sub, target_sub, random_seed=random_seed, control=None
                     )
                 except Exception as e:
                     print(f"    Error computing local shareSpace: {e}")
                     # Fallback?
                     pass
             else:
                 print("    Warning: Too few common genes for local shareSpace.")

        if 'shareSpace' in source_sub.obsm and 'shareSpace' in target_sub.obsm:
            M = cosine_distance_tensor(source_sub.obsm['shareSpace'], target_sub.obsm['shareSpace'])
            M = M.to(device)
        else:
            # Fallback M? Or Error?
            # scFUGW needs M.
            raise ValueError("shareSpace not available for batch.")
        
        # Run scFUGW
        pi = scFUGW_RNA_Spatial_with_cost(
            target=target_sub,
            source=source_sub,
            target_cost='cost_matrix',
            source_cost='cost_matrix',
            M=M,
            alpha=alpha,
            rho=rho,
            eps=eps,
            random_seed=random_seed
        )
        
        if hasattr(pi, 'cpu'):
            pi = pi.cpu().numpy()
            
        # Project requested keys
        row_sums = pi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        
        for key in projected_results.keys():
            if key in target_sub.obsm:
                t_data = target_sub.obsm[key]
            elif key == 'X':
                t_data = target_sub.X
                if issparse(t_data): t_data = t_data.toarray()
            
            mapped_batch = pi @ t_data
            mapped_batch = mapped_batch / row_sums
            
            projected_results[key][s_idxs] = mapped_batch
        
        # Label transfer within this batch
        if label_columns:
            MT = target_sub.obsm[tgt_feat_key]
            MP = pi @ MT
            MP = MP / row_sums
            
            batch_result = {'source_idx': s_idxs}
            for col in label_columns:
                batch_result[col] = label_transfer(MP, MT, target_sub.obs[col], top=top_k)
            all_batch_label_results.append(pd.DataFrame(batch_result))
            
            if verbose:
                print(f"    Label transfer done for batch {i+1}")
        
    # Merge label transfer results across all batches
    label_results_df = None
    if label_columns and all_batch_label_results:
        label_results_df = pd.concat(all_batch_label_results, ignore_index=True)
        label_results_df = label_results_df.sort_values('source_idx').reset_index(drop=True)
        label_results_df['spot_id'] = source_adata.obs_names[label_results_df['source_idx'].values]
        label_results_df = label_results_df.drop(columns=['source_idx'])
        cols = ['spot_id'] + label_columns
        label_results_df = label_results_df[cols]

    # Return results
    if tgt_feat_key in target_adata.obsm:
        full_target_features = target_adata.obsm[tgt_feat_key]
    else:
        full_target_features = target_adata.X
        if issparse(full_target_features): full_target_features = full_target_features.toarray()
        
    return label_results_df, projected_results, full_target_features
