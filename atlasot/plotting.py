"""Minimal spatial visualization functions for AtlasOT results.

All plotting functions expect coordinates in **matplotlib convention**
(origin bottom-left, Y increases upward).  If your coordinates come from
Visium (origin top-left, Y increases downward), call :func:`flip_visium_y`
first — see the P017 script for a complete worked example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # no extra types needed — all inputs are numpy/pandas


def flip_visium_y(coords: np.ndarray) -> np.ndarray:
    """Flip Y-axis for Visium-style pixel coordinates (origin top-left → bottom-left).

    Visium stores coordinates with Y=0 at the **top** of the tissue image.
    Matplotlib draws Y=0 at the **bottom**.  This function mirrors the
    Y-axis so the tissue appears right-side up in matplotlib.

    Parameters
    ----------
    coords : np.ndarray or pd.DataFrame
        Spatial coordinates, shape (n_spots, 2), column 1 = Y.

    Returns
    -------
    np.ndarray
        Coordinates with Y flipped (``max(Y) - Y``).
    """
    if isinstance(coords, pd.DataFrame):
        coords = coords.values
    c = np.asarray(coords, dtype=float).copy()
    c[:, 1] = c[:, 1].max() - c[:, 1]
    return c


def _compute_proportions(
    pi: np.ndarray,
    source_labels: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-spot cell type proportions from transport plan.

    Returns (prop, type_names) where prop has shape (n_types, n_spots)
    and is normalized so each column sums to 1.
    """
    one_hot = pd.get_dummies(source_labels.astype(str))
    type_names = list(one_hot.columns)
    prop = (one_hot.values.astype(float).T @ pi)
    col_sum = prop.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    return prop / col_sum, type_names


def spatial_heatmap(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    gene_names: list[str] | None = None,
    title: str = '',
    cmap: str = 'viridis',
    spot_size: int = 10,
    save: str | None = None,
) -> plt.Figure:
    """Plot gene expression on spatial coordinates, colored by percentile rank.

    Values are always converted to percentile ranks (0–1) so the colour
    scale is robust to outliers and comparable across genes — matching the
    convention used in P017-Moran-Quality.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates in matplotlib convention (origin bottom-left,
        Y increases upward).  Use :func:`flip_visium_y` first if your
        coordinates came from Visium.  Shape (n_spots, 2).
    values : np.ndarray
        Values per spot, shape (n_spots,) for a single gene, or
        (n_spots, n_genes) for multiple genes (facet grid).
    gene_names : list of str, optional
        Gene names for subplot titles.  Auto-generated if omitted.
    title : str
        Figure-level title.
    cmap : str
        Matplotlib colormap name.
    spot_size : int
        Scatter dot size.
    save : str or None
        If given, saves figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib Figure — auto-displays in Jupyter notebooks.
    """
    if isinstance(coords, pd.DataFrame):
        coords = coords.values
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_genes = values.shape[1]
    if n_genes > 1 and gene_names is None:
        gene_names = [f'Gene {i + 1}' for i in range(n_genes)]

    if n_genes == 1:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        axes = [ax]
    else:
        ncols = min(4, n_genes)
        nrows = int(np.ceil(n_genes / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), dpi=150)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(n_genes):
        ax = axes[i]
        # Always percentile-rank — robust to outliers, comparable across genes
        vals = rankdata(values[:, i]) / len(values)
        im = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=spot_size,
                        cmap=cmap, vmin=0.0, vmax=1.0, edgecolors='none')
        ax.set_title(gene_names[i] if gene_names else '', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7)

    for j in range(n_genes, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=150)
    return fig


def spatial_deconvolution(
    pi: np.ndarray,
    source_labels: pd.Series,
    coords: np.ndarray,
    *,
    spot_size: int = 10,
    save: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot per-cell-type proportion on tissue (facet grid).

    Computes type proportions as ``onehot(labels).T @ pi``, normalized
    per spot so each column sums to 1.

    Parameters
    ----------
    pi : np.ndarray
        Transport plan, shape (n_source, n_target).
    source_labels : pd.Series
        Cell type labels for each source cell.
    coords : np.ndarray
        Spatial coordinates of target spots in matplotlib convention
        (origin bottom-left, Y increases upward).  Use
        :func:`flip_visium_y` first for Visium coordinates.
        Shape (n_target, 2).
    spot_size : int
        Scatter dot size.
    save : str or None
        If given, saves figure to this path.

    Returns
    -------
    fig : plt.Figure
        Facet grid of per-type spatial proportions.
    prop_df : pd.DataFrame
        Proportion matrix of shape (n_spots, n_types).
    """
    if isinstance(coords, pd.DataFrame):
        coords = coords.values
    coords = np.asarray(coords, dtype=float)
    prop, type_names = _compute_proportions(pi, source_labels)

    n_types = len(type_names)
    ncols = min(4, n_types)
    nrows = int(np.ceil(n_types / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), dpi=150)
    axes = axes.flatten() if n_types > 1 else [axes]

    for i, ct in enumerate(type_names):
        ax = axes[i]
        vals = prop[i]
        vmax_ct = max(float(np.percentile(vals, 99)), 0.01)
        im = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=spot_size,
                        cmap='plasma', vmin=0, vmax=vmax_ct, edgecolors='none')
        ax.set_title(ct, fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)

    for j in range(n_types, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Spatial Deconvolution', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=150)

    return fig, pd.DataFrame(prop.T, columns=type_names)


def dominant_type_map(
    pi: np.ndarray,
    source_labels: pd.Series,
    coords: np.ndarray,
    *,
    spot_size: int = 12,
    save: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot dominant cell type per spatial spot.

    Each spot is colored by the cell type with the highest transport mass.

    Parameters
    ----------
    pi : np.ndarray
        Transport plan, shape (n_source, n_target).
    source_labels : pd.Series
        Cell type labels for each source cell.
    coords : np.ndarray
        Spatial coordinates of target spots in matplotlib convention
        (origin bottom-left, Y increases upward).  Use
        :func:`flip_visium_y` first for Visium coordinates.
        Shape (n_target, 2).
    spot_size : int
        Scatter dot size.
    save : str or None
        If given, saves figure to this path.

    Returns
    -------
    fig : plt.Figure
        Dominant type map with legend.
    prop_df : pd.DataFrame
        Proportion matrix of shape (n_spots, n_types).
    """
    if isinstance(coords, pd.DataFrame):
        coords = coords.values
    coords = np.asarray(coords, dtype=float)
    prop, type_names = _compute_proportions(pi, source_labels)
    n_types = len(type_names)

    dominant_idx = np.argmax(prop, axis=0)

    cmap = plt.cm.tab20 if n_types <= 20 else plt.cm.tab20b
    colors = [cmap(i / n_types) for i in range(n_types)]

    type_counts = np.bincount(dominant_idx, minlength=n_types)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for i in range(n_types):
        mask = dominant_idx == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], s=spot_size,
                   label=f'{type_names[i]} ({type_counts[i]})', edgecolors='none')

    ax.set_title('Dominant Cell Type', fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8,
              title='Cell Type (n spots)', title_fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=150)

    return fig, pd.DataFrame(prop.T, columns=type_names)
