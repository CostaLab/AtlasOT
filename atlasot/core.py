import warnings
import torch
import random
from fugw.mappings import FUGW
from .evaluate import *


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)



def scFUGW_RNA_Spatial_with_cost(
        target,
        source,
        # target_reduction = 'RNA_pca_l2_norm',
        target_cost = 'cost_matrix',
        source_cost = 'cost_matrix',
        alpha=0.6,
        rho=1.1,
        eps=1e-1,
        lambda_laplacian = 5.0,
        random_seed=3407,
        M = None,
        L = None,
):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)


    # target.var.index = target.var.index.str.split(':', expand=True)
    # target.var['features'] = target.var.index

    target_shareSpace = torch.from_numpy(target.obsm['shareSpace'])
    source_shareSpace = torch.from_numpy(source.obsm['shareSpace'])

    target_shareSpace = target_shareSpace / torch.linalg.norm(
        target_shareSpace, dim=1
    ).reshape(-1, 1)
    source_shareSpace = source_shareSpace / torch.linalg.norm(
        source_shareSpace, dim=1
    ).reshape(-1, 1)


    target_geometry = torch.tensor(target.obsp[target_cost], dtype=torch.float32)
    source_geometry = torch.tensor(source.obsp[source_cost], dtype=torch.float32)

    target_geometry = target_geometry / target_geometry.max()
    source_geometry = source_geometry / source_geometry.max()

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps, lambda_laplacian=lambda_laplacian)


    _ = mapping.fit(
        source_shareSpace.T,
        target_shareSpace.T,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        solver="sinkhorn",
        verbose=True,
        M = M,
        L = L,
    )

    pi = mapping.pi

    # mapped_data = np.dot(pi, target.obsm[target_reduction]) / pi.sum(dim=1).reshape(-1, 1)
    return pi

def scFUGW_RNA_Spatial_with_cost_distribution(
        target,
        source,
        # target_reduction = 'RNA_pca_l2_norm',
        target_cost = 'cost_matrix',
        source_cost = 'cost_matrix',
        source_weights = None,
        target_weights = None,
        alpha=0.1,
        rho=1.1,
        eps=1e-5,
        lambda_laplacian = 0.1,
        random_seed=3407,
        M = None,
        L = None,
):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)


    # target.var.index = target.var.index.str.split(':', expand=True)
    # target.var['features'] = target.var.index

    target_shareSpace = torch.from_numpy(target.obsm['shareSpace'])
    source_shareSpace = torch.from_numpy(source.obsm['shareSpace'])

    target_shareSpace = target_shareSpace / torch.linalg.norm(
        target_shareSpace, dim=1
    ).reshape(-1, 1)
    source_shareSpace = source_shareSpace / torch.linalg.norm(
        source_shareSpace, dim=1
    ).reshape(-1, 1)



    # target.obsm[target_reduction] = target.obsm[target_reduction] / torch.linalg.norm(
    #     torch.from_numpy(target.obsm[target_reduction]), dim=1
    # ).reshape(-1, 1)
    # source.obsm[source_reduction] = source.obsm[source_reduction] / torch.linalg.norm(
    #     torch.from_numpy(source.obsm[source_reduction]), dim=1
    # ).reshape(-1, 1)

    # target_geometry = torch.cdist(target.obsm[target_reduction], target.obsm[target_reduction])
    # source_geometry = torch.cdist(source.obsm[source_reduction], source.obsm[source_reduction])

    target_geometry = torch.tensor(target.obsp[target_cost], dtype=torch.float32)
    source_geometry = torch.tensor(source.obsp[source_cost], dtype=torch.float32)

    target_geometry = target_geometry / target_geometry.max()
    source_geometry = source_geometry / source_geometry.max()

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps, lambda_laplacian=lambda_laplacian)


    _ = mapping.fit(
        source_shareSpace.T,
        target_shareSpace.T,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        source_weights=source_weights,
        target_weights=target_weights,
        solver="sinkhorn",
        verbose=True,
        M = M,
        L = L,
    )

    pi = mapping.pi

    # mapped_data = np.dot(pi, target.obsm[target_reduction]) / pi.sum(dim=1).reshape(-1, 1)
    return pi



def scFUGW_RNA_Spatial(
        target,
        source,
        target_reduction='RNA_pca_l2_norm',
        source_reduction='scopen',
        alpha=0.1,
        rho=1.1,
        eps=1e-5,
        random_seed=3407,
        M = None,
):

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)


    target.var.index = target.var.index.str.split(':', expand=True)
    target.var['features'] = target.var.index

    target.obsm['shareSpace'] = torch.from_numpy(target.obsm['shareSpace'])
    source.obsm['shareSpace'] = torch.from_numpy(source.obsm['shareSpace'])

    target.obsm['shareSpace'] = target.obsm['shareSpace'] / torch.linalg.norm(
        target.obsm['shareSpace'], dim=1
    ).reshape(-1, 1)
    source.obsm['shareSpace'] = source.obsm['shareSpace'] / torch.linalg.norm(
        source.obsm['shareSpace'], dim=1
    ).reshape(-1, 1)

    target.obsm[target_reduction] = target.obsm[target_reduction] / torch.linalg.norm(
        torch.from_numpy(target.obsm[target_reduction]), dim=1
    ).reshape(-1, 1)
    source.obsm[source_reduction] = source.obsm[source_reduction] / torch.linalg.norm(
        torch.from_numpy(source.obsm[source_reduction]), dim=1
    ).reshape(-1, 1)

    target_geometry = torch.cdist(target.obsm[target_reduction], target.obsm[target_reduction])
    source_geometry = torch.cdist(source.obsm[source_reduction], source.obsm[source_reduction])

    target_geometry = target_geometry / target_geometry.max()
    source_geometry = source_geometry / source_geometry.max()

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps)


    _ = mapping.fit(
        source.obsm['shareSpace'].T,
        target.obsm['shareSpace'].T,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        solver="sinkhorn",
        verbose=True,
        M = M,
    )

    pi = mapping.pi

    mapped_data = np.dot(pi, target.obsm[target_reduction]) / pi.sum(dim=1).reshape(-1, 1)
    return mapped_data, pi











def scFUGW(
        rna,
        atac_peaks,
        rna_reduction='RNA_pca_l2_norm',
        atac_reduction='scopen',
        mapping_direction = 'ATAC2RNA',
        cell_type = 'cell_type',
        alpha=0.1,
        rho=1.1,
        eps=1e-5,
        random_seed=3407,
        device=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rna.var.index = rna.var.index.str.split(':', expand=True)
    rna.var['features'] = rna.var.index

    # Convert to torch tensors and move to device
    rna.obsm['shareSpace'] = torch.from_numpy(rna.obsm['shareSpace']).to(device).float()
    atac_peaks.obsm['shareSpace'] = torch.from_numpy(atac_peaks.obsm['shareSpace']).to(device).float()

    rna.obsm['shareSpace_norm'] = rna.obsm['shareSpace'] / torch.linalg.norm(
        rna.obsm['shareSpace'], dim=1
    ).reshape(-1, 1)
    atac_peaks.obsm['shareSpace_norm'] = atac_peaks.obsm['shareSpace'] / torch.linalg.norm(
        atac_peaks.obsm['shareSpace'], dim=1
    ).reshape(-1, 1)

    # Convert reduction data to torch tensors and move to device
    rna_reduction_data = torch.from_numpy(rna.obsm[rna_reduction]).to(device).float()
    atac_reduction_data = torch.from_numpy(atac_peaks.obsm[atac_reduction]).to(device).float()
    
    rna.obsm[rna_reduction] = rna_reduction_data / torch.linalg.norm(
        rna_reduction_data, dim=1
    ).reshape(-1, 1)
    atac_peaks.obsm[atac_reduction] = atac_reduction_data / torch.linalg.norm(
        atac_reduction_data, dim=1
    ).reshape(-1, 1)

    rna_geometry = torch.cdist(rna.obsm[rna_reduction], rna.obsm[rna_reduction])
    atac_geometry = torch.cdist(atac_peaks.obsm[atac_reduction], atac_peaks.obsm[atac_reduction])

    rna_geometry = rna_geometry / rna_geometry.max()
    atac_geometry = atac_geometry / atac_geometry.max()

    mapping = FUGW(alpha=alpha, rho=rho, eps=eps)

    if mapping_direction == 'ATAC2RNA':
        _ = mapping.fit(
            atac_peaks.obsm['shareSpace_norm'].T,
            rna.obsm['shareSpace_norm'].T,
            source_geometry=atac_geometry,
            target_geometry=rna_geometry,
            solver="sinkhorn",
            verbose=True,
        )

        pi = mapping.pi

        mapped_data = torch.matmul(pi, rna.obsm[rna_reduction]) / pi.sum(dim=1).reshape(-1, 1)
        
        # Convert back to CPU numpy if device is GPU
        if device.type == 'cuda':
            mapped_data = mapped_data.cpu().numpy()
            pi = pi.cpu()
        
        return mapped_data, pi

    elif mapping_direction == 'RNA2ATAC':
        _ = mapping.fit(
            rna.obsm['shareSpace_norm'].T,
            atac_peaks.obsm['shareSpace_norm'].T,
            source_geometry=rna_geometry,
            target_geometry=atac_geometry,
            solver="sinkhorn",
            verbose=True,
        )
        pi = mapping.pi

        mapped_data = torch.matmul(pi, atac_peaks.obsm[atac_reduction]) / pi.sum(dim=1).reshape(-1, 1)
        
        # Convert back to CPU numpy if device is GPU
        if device.type == 'cuda':
            mapped_data = mapped_data.cpu().numpy()
            pi = pi.cpu()
        
        return mapped_data, pi

    else:
        raise NotImplementedError("Please set mapping_direction to be 'ATAC2RNA' or 'RNA2ATAC'.")
