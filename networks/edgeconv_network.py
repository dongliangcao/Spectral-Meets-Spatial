import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv

from utils.registry import NETWORK_REGISTRY


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def get_edge_index(face):
    edge_index_one = torch.cat(
        (face[:, [0, 1]], face[:, [0, 2]], face[:, [1, 2]]), 0).t()
    # undirected edge
    edge_index = torch.zeros([2, edge_index_one.shape[1] * 2], dtype=face.dtype, device=face.device)
    edge_index[:, :edge_index_one.shape[1]] = edge_index_one
    edge_index[0, edge_index_one.shape[1]:] = edge_index_one[1, :]
    edge_index[1, edge_index_one.shape[1]:] = edge_index_one[0, :]
    return edge_index


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


@NETWORK_REGISTRY.register()
class ResnetECPos(torch.nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, use_mlp=False):
        super(ResnetECPos, self).__init__()
        self.c_dim = c_dim

        self.fc_pos = torch.nn.Linear(dim, 2*hidden_dim)
        self.block_0 = EdgeConv(ResnetBlockFC(4*hidden_dim, hidden_dim))
        self.block_1 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_2 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_3 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        self.block_4 = EdgeConv(ResnetBlockFC(4*hidden_dim+2*dim, hidden_dim))
        if use_mlp:
            self.fc_c = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, c_dim)
            )
        else:
            self.fc_c = torch.nn.Linear(hidden_dim, c_dim)

        self.actvn = torch.nn.ReLU()
        self.pool = maxpool

    def forward(self, verts, faces, feats=None):
        squeeze = False
        if verts.dim() == 3:
            verts, faces = verts.squeeze(0), faces.squeeze(0)
            if feats is not None:
                feats = feats.squeeze(0)
            squeeze = True
        edge_index = get_edge_index(faces)
        p = verts if feats is None else feats
        net = self.fc_pos(p)
        net = self.block_0(net, edge_index)

        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_1(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_2(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_3(net, edge_index)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled, p], dim=1)

        net = self.block_4(net, edge_index)

        c = self.fc_c(self.actvn(net))

        if squeeze:
            c = c.unsqueeze(0)
        return c
