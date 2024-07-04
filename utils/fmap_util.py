import numpy as np
import scipy.spatial

import torch


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p


def knn_query(feat_x, feat_y, k, dim=-2):
    dist = torch.cdist(feat_x, feat_y)
    indices = torch.topk(-dist, k=k, dim=dim).indices
    return indices


def knn(x, y, k):
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2
    assert y.dim() == 2
    assert x.size(1) == y.size(1)

    # Rescale x and y.
    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    tree = scipy.spatial.cKDTree(x.detach().cpu().numpy())
    dist, col = tree.query(y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = ~torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)


def knn_graph(x, k, loop=False, flow='source_to_target'):
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = knn(x, x, k if loop else k + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


def refine_pointmap_icp(p2p, evecs_x, evecs_y, n_iters=10):
    """
    Regular Iterative Closest Point (ICP) to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        n_iters (int, optional): number of iterations. Default 10.
    """
    p2p_refined = p2p
    for _ in range(n_iters):
        C21_refined = pointmap2fmap(p2p_refined, evecs_x, evecs_y)
        p2p_refined = fmap2pointmap(C21_refined, evecs_y, evecs_x)

    return p2p_refined


def refine_pointmap_zoomout(p2p, evecs_x, evecs_y, k_start, step=1):
    """
    ZoomOut to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        k_start (int): number of eigenvectors to start
        step (int, optional): step size. Default 1.
    """
    k_end = evecs_x.shape[1]
    inds = np.arange(k_start, k_end + step, step)

    p2p_refined = p2p
    for i in inds:
        C21_refined = pointmap2fmap(p2p_refined, evecs_x[:, :i], evecs_y[:, :i])
        p2p_refined = fmap2pointmap(C21_refined, evecs_y[:, :i], evecs_x[:, :i])

    return p2p_refined
