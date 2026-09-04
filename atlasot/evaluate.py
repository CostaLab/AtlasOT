from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st
from anndata import AnnData



def find_threshold(
    values: np.ndarray,
    threshold: float = 1e-3,
) -> int | None:
    """Find the first index where a value drops below the given threshold.

    Used to determine PCA component count from explained variance ratios.

    Parameters
    ----------
    values : np.ndarray
        Monotonically decreasing array (e.g., explained variance ratios).
    threshold : float
        Cutoff value.

    Returns
    -------
    int or None
        Index of the first value below threshold, or None if all are above.
    """
    for i, v in enumerate(values):
        if v < threshold:
            # print(i, v)
            return i



def get_acc(
    MP: np.ndarray,
    MT: np.ndarray,
    mappedData: np.ndarray,
    truthData: pd.Series,
    top: int = 1,
) -> float:
    """Evaluate label transfer accuracy via KNN voting.

    For each mapped point, finds its top-k nearest neighbors in the target
    space and checks if the predicted label matches the majority vote.

    Parameters
    ----------
    MP : np.ndarray
        Mapped source coordinates of shape (n_source, n_features).
    MT : np.ndarray
        Target coordinates of shape (n_target, n_features).
    mappedData : np.ndarray
        Predicted labels for source cells.
    truthData : pd.Series
        Ground-truth labels for target cells.
    top : int
        Number of nearest neighbors for voting.

    Returns
    -------
    float
        Accuracy score in [0, 1].
    """
    length = len(MP)
    counts = 0

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        if mappedData[i] == truthData[indexes].value_counts().index[0]:
            counts += 1
    return counts / length



def label_transfer(
    MP: np.ndarray,
    MT: np.ndarray,
    truthData: pd.Series,
    top: int = 1,
) -> list:
    """Transfer labels from target to source via KNN voting in embedding space.

    Parameters
    ----------
    MP : np.ndarray
        Mapped source coordinates of shape (n_source, n_features).
    MT : np.ndarray
        Target coordinates of shape (n_target, n_features).
    truthData : pd.Series
        Labels for target cells.
    top : int
        Number of nearest neighbors for voting.

    Returns
    -------
    list
        Predicted labels for each source cell.
    """
    predicted_label = []

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        predicted_label.append(truthData[indexes].value_counts().index[0])

    return predicted_label


def ot_label_transfer(
    pi: np.ndarray,
    target_labels: pd.Series,
) -> np.ndarray:
    """Transfer labels via optimal transport coupling matrix.

    Uses the transport plan π to map one-hot encoded target labels to source
    cells, then takes argmax to produce hard label predictions.

    Parameters
    ----------
    pi : np.ndarray
        Transport matrix of shape (n_source, n_target).
    target_labels : pd.Series
        Categorical labels for target cells.

    Returns
    -------
    np.ndarray
        Predicted labels for each source cell.
    """
    one_hot = pd.get_dummies(target_labels)
    classes = one_hot.columns.values
    label_probs = pi @ one_hot.values.astype(float)
    pred_labels = classes[np.argmax(label_probs, axis=1)]
    return pred_labels


def gene_imputation(
    pi: np.ndarray,
    source_adata: AnnData,
) -> pd.DataFrame:
    """Impute gene expression from source to target via transport plan.

    Computes target_expression = source_expression.T @ π.

    Parameters
    ----------
    pi : np.ndarray
        Transport matrix of shape (n_source, n_target).
    source_adata : AnnData
        Source AnnData with gene expression in ``.X``.

    Returns
    -------
    pd.DataFrame
        Imputed gene expression of shape (n_target, n_genes),
        with gene names as columns.
    """
    from scipy.sparse import csr_matrix, issparse

    X = source_adata.X.toarray() if issparse(source_adata.X) else np.asarray(source_adata.X)
    counts = csr_matrix(X)
    imputed = counts.T.dot(pi).T

    return pd.DataFrame(imputed, columns=source_adata.var_names)


def map_to_target(
    pi: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    """Map source cells to target feature space via transport plan.

    Computes :math:`S_{mapped} = \\pi \\cdot T / \\sum(\\pi)`, i.e. each source cell
    becomes a weighted average of target cells according to the transport plan.

    To map in the reverse direction (target → source), pass ``pi.T`` and the
    source feature matrix instead: ``map_to_target(pi.T, source_features)``.

    Parameters
    ----------
    pi : np.ndarray
        Transport matrix of shape (n_source, n_target).
    target_features : np.ndarray
        Target feature matrix of shape (n_target, n_features).

    Returns
    -------
    np.ndarray
        Mapped coordinates of shape (n_source, n_features).
    """
    row_sum = pi.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (pi @ target_features) / row_sum


def PCC(raw: pd.DataFrame, impute: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation coefficient per gene."""
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                pearsonr = 0
            else:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                pearsonr, _ = st.pearsonr(raw_col,impute_col)
            pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[label])
            result = pd.concat([result, pearson_df],axis=1)
    else:
        raise ValueError(
            f"PCC: row count mismatch — raw has {raw.shape[0]} rows, "
            f"impute has {impute.shape[0]} rows"
        )
    return result


def scale_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to sum to 1."""
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def JS(
    raw: pd.DataFrame,
    impute: pd.DataFrame,
    scale: str = 'scale_plus',
) -> pd.DataFrame:
    """Compute Jensen-Shannon divergence per gene."""

    if scale == 'scale_plus':
        raw = scale_plus(raw)
        impute = scale_plus(impute)
    else:
        print(f"JS: unknown scale='{scale}', falling back to raw values (no normalization)")

    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                JS = 1
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                raw_col = raw_col.fillna(1e-20)
                impute_col = impute_col.fillna(1e-20)
                M = (raw_col + impute_col) / 2
                JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
            JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
            result = pd.concat([result, JS_df], axis=1)
    else:
        raise ValueError(
            f"JS: row count mismatch — raw has {raw.shape[0]} rows, "
            f"impute has {impute.shape[0]} rows"
        )
    return result

def scale_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize columns."""
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result


def RMSE(
    raw: pd.DataFrame,
    impute: pd.DataFrame,
    scale: str = 'zscore',
) -> pd.DataFrame:
    """Compute root mean squared error per gene."""
    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print(f"RMSE: unknown scale='{scale}', falling back to raw values (no normalization)")
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                RMSE = 1.5
            else:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
            result = pd.concat([result, RMSE_df],axis=1)
    else:
        raise ValueError(
            f"RMSE: row count mismatch — raw has {raw.shape[0]} rows, "
            f"impute has {impute.shape[0]} rows"
        )
    return result


def cal_ssim(
    im1: np.ndarray,
    im2: np.ndarray,
    M: float,
) -> float:
    """Compute structural similarity index between two 1D signals."""
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to [0, 1] by dividing by the maximum."""
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def SSIM(
    raw: pd.DataFrame,
    impute: pd.DataFrame,
    scale: str = 'scale_max',
) -> pd.DataFrame:
    """Compute structural similarity index per gene."""
    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print(f"SSIM: unknown scale='{scale}', falling back to raw values (no normalization)")
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                ssim = 0
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                M = max(raw_col.max(), impute_col.max())
                # [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                raw_col_2 = np.array(raw_col)
                raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                impute_col_2 = np.array(impute_col)
                impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                ssim = cal_ssim(raw_col_2, impute_col_2, M)

            ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
            result = pd.concat([result, ssim_df], axis=1)
    else:
        raise ValueError(
            f"SSIM: row count mismatch — raw has {raw.shape[0]} rows, "
            f"impute has {impute.shape[0]} rows"
        )
    return result


def compute_all_evaluation(
    raw: pd.DataFrame,
    impute: pd.DataFrame,
) -> None:
    """Print all four evaluation metrics (PCC, JS, RMSE, SSIM)."""
    pcc_result = PCC(raw, impute)
    js_result = JS(raw, impute)
    zscore_result = RMSE(raw, impute)
    SSIM_result = SSIM(raw, impute)

    print('PCC: ', pcc_result.mean().mean())
    print('JS: ', js_result.mean().mean())
    print('RMSE: ', zscore_result.mean().mean())
    print('SSIM: ', SSIM_result.mean().mean())
