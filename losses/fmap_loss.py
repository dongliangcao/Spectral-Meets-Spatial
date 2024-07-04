import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0, return_mean=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.return_mean = return_mean

    def forward(self, a, b=0):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        if self.return_mean:
            return self.loss_weight * torch.mean(loss)
        else:
            return self.loss_weight * torch.sum(loss)

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = F.mse_loss(a, b)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap

    def forward(self, C12, C21, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = SquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        losses = dict()
        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) + criterion(torch.bmm(C21, C12), eye_batch)
            bijectivity_loss *= self.w_bij
            losses['l_bij'] = bijectivity_loss

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) + \
                                 criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)
            orthogonality_loss *= self.w_orth
            losses['l_orth'] = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                       torch.einsum('ab,abc->abc', evals_2, C12))
            laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                        torch.einsum('ab,abc->abc', evals_1, C21))
            laplacian_loss *= self.w_lap
            losses['l_lap'] = laplacian_loss

        return losses
