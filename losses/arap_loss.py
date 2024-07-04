import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY


def get_neigh(face):
    return torch.cat((face[:, [0, 1]], face[:, [0, 2]], face[:, [1, 2]]), dim=0)


def arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert, L):
    device = vert_diff_t.device
    S_neigh = torch.bmm(vert_diff_t.unsqueeze(2), vert_diff_0.unsqueeze(1))

    if L is not None:
        weight = -L[neigh[:, 0], neigh[:, 1]]
        S_neigh = weight[:, None, None] * S_neigh

    S = torch.zeros(n_vert, 3, 3, device=device)

    S = torch.index_add(S, 0, neigh[:, 0], S_neigh)
    S = torch.index_add(S, 0, neigh[:, 1], S_neigh)

    U, _, V = torch.svd(S.cpu(), compute_uv=True)

    U = U.to(device)
    V = V.to(device)

    R = torch.bmm(U, V.transpose(1, 2))

    Sigma = torch.ones(R.shape[0], 1, 3, device=device)
    Sigma[:, :, 2] = torch.det(R).unsqueeze(1)

    R = torch.bmm(U * Sigma, V.transpose(1, 2))

    return R


def arap_energy_exact(vert_t, vert_0, neigh, L=None, lambda_reg_len=1e-6):
    n_vert = vert_t.shape[0]

    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

    R_t = arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert, L)

    R_neigh_t = 0.5 * (
        torch.index_select(R_t, 0, neigh[:, 0])
        + torch.index_select(R_t, 0, neigh[:, 1])
    )

    vert_diff_0_rot = torch.bmm(R_neigh_t, vert_diff_0.unsqueeze(2)).squeeze()
    acc_t_neigh = vert_diff_t - vert_diff_0_rot

    E_arap = acc_t_neigh.norm() ** 2 + lambda_reg_len * (vert_t - vert_0).norm() ** 2

    return E_arap


@LOSS_REGISTRY.register()
class ArapLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ArapLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, vert_t, vert_0, neigh, L=None):
        if L is not None:
            l_arap = arap_energy_exact(vert_t, vert_0, neigh, L)
        else:
            l_arap = arap_energy_exact(vert_t, vert_0, neigh)
        return self.loss_weight * l_arap
