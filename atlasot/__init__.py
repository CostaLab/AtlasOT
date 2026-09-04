from .process_data import (
    preprocess_atac_peaks,
    reduction_atac_peaks,
    preprocess_atac_genes,
    reduction_atac_genes,
    preprocess_rna,
    reduction_rna,
    preprocess_spatial,
    reduction_spatial,
    find_shared_space,
)
from .cost import (
    cosine_distance_tensor,
    compute_knn_normalized_laplacian,
    compute_geodesic_distance,
    compute_spatial_geodesic,
    graph_smooth_results,
)
from .core import scFUGW_RNA_Spatial_with_cost
from .evaluate import (
    find_threshold,
    get_acc,
    label_transfer,
    ot_label_transfer,
    gene_imputation,
    map_to_target,
    PCC,
    scale_plus,
    JS,
    scale_z_score,
    RMSE,
    cal_ssim,
    scale_max,
    SSIM,
    compute_all_evaluation,
)
from .minibatch import (
    split_indices,
    compute_batch_cost_matrix,
    minibatch_atlasot,
)
from .plotting import (
    flip_visium_y,
    spatial_heatmap,
    spatial_deconvolution,
    dominant_type_map,
)
from .tune import sweep_atlasot_alpha_eps

def __getattr__(name: str):
    """Lazily expose the scOpen entry point.

    ``import atlasot`` must not require scopen (RNA/Spatial-only users never
    touch it); the real import happens on first use inside
    :func:`reduction_atac_peaks`. This PEP 562 hook keeps
    ``atlasot.scopen_dr`` working for any code that imported it from the
    package namespace.
    """
    if name == "scopen_dr":
        from scopen.Main import scopen_dr
        return scopen_dr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.3.0"
