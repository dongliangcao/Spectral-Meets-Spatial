import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.logger import get_root_logger
from utils.tensor_util import to_device, to_numpy
from utils.fmap_util import nn_query, fmap2pointmap
from utils.vector_adam import VectorAdam


def get_neigh(face):
    return torch.cat((face[:, [0, 1]], face[:, [0, 2]], face[:, [1, 2]]), dim=0)


def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)


class DirichletLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, feats, L):
        de = cdot(feats, torch.mm(L, feats), dim=0)
        loss = torch.mean(de)

        return self.loss_weight * loss


@torch.set_grad_enabled(True)
def compute_deformation(data_x, data_y, n_iter, dx=None):
    vert_x, face_x = data_x['verts'].squeeze(0), data_x['faces'].squeeze(0)
    vert_y, face_y = data_y['verts'].squeeze(0), data_y['faces'].squeeze(0)
    mesh_x = Meshes(verts=[vert_x], faces=[face_x])
    mesh_y = Meshes(verts=[vert_y], faces=[face_y])

    # get laplacian matrix
    Lx, Ly = data_x['L'].squeeze(0), data_y['L'].squeeze(0)

    # align shape x to shape y
    if dx is None:
        dx = torch.zeros_like(vert_x, requires_grad=True)
    else:
        dx = dx.clone()
        dx.requires_grad = True
    optimizer = VectorAdam(params=[dx], lr=5e-4, axis=-1)

    w_chamfer = 50.0
    w_dirichlet = 0.1
    dirichlet_loss = DirichletLoss(loss_weight=w_dirichlet)
    for n in range(1, n_iter + 1):
        mesh_x_align = mesh_x.offset_verts(dx)

        # compute chamfer distance loss
        sample_x = sample_points_from_meshes(mesh_x_align, 40000)
        sample_y = sample_points_from_meshes(mesh_y, 40000)
        loss_chamfer, _ = chamfer_distance(sample_x, sample_y)
        loss_chamfer = w_chamfer * loss_chamfer

        # compute dirichlet energy loss
        loss_dirichlet = w_dirichlet * dirichlet_loss(dx, Lx)

        loss = loss_chamfer + loss_dirichlet

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save final result
    vert_x_align = vert_x + dx

    return vert_x_align


@MODEL_REGISTRY.register()
class SpectralMeetsSpatialModel(BaseModel):
    def __init__(self, opt):
        self.pose_timestep = opt.get('pose_timestep', 0)
        self.pose_milestones = opt.get('pose_milestones', None)
        self.optim_iter = opt.get('optim_iter', 1000)
        self.pose_alignment = opt.get('pose_alignment', False)
        self.non_isometric = opt.get('non-isometric', False)
        super(SpectralMeetsSpatialModel, self).__init__(opt)

    def update_model_per_epoch(self):
        super(SpectralMeetsSpatialModel, self).update_model_per_epoch()
        if self.pose_milestones is not None and self.curr_epoch in self.pose_milestones:
            logger = get_root_logger()
            logger.info(f'Update number of pose steps from '
                        f'{self.pose_timestep} to {self.pose_milestones[self.curr_epoch]}')
            self.pose_timestep = self.pose_milestones[self.curr_epoch]

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
        assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'

        # extract feature
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)  # [B, Nx, Ny], [B, Ny, Nx]

        # compute functional map related loss
        if 'surfmnet_loss' in self.losses:
            self.compute_fmap_loss(data_x, data_y, feat_x, feat_y, Pxy, Pyx)

        # Interpolation
        Pxy, Pyx = Pxy.squeeze(0), Pyx.squeeze(0)
        vert_x, vert_y = data_x['verts'].squeeze(0), data_y['verts'].squeeze(0)
        face_x, face_y = data_x['faces'].squeeze(0), data_y['faces'].squeeze(0)

        # from shape x to shape y
        vert_x_pred_arr = self.compute_displacement(vert_x, vert_y, face_x, Pxy)  # [n_vert_x, 3, T+1]

        # from shape y to shape x
        vert_y_pred_arr = self.compute_displacement(vert_y, vert_x, face_y, Pyx)  # [n_vert_y, 3, T+1]

        # compute pose interpolation related loss
        if 'arap_loss' in self.losses:
            self.compute_pose_interpolation_loss(vert_x, vert_x_pred_arr, face_x)
            self.compute_pose_interpolation_loss(vert_y, vert_y_pred_arr, face_y)

        # compute alignment loss
        vert_y_1 = Pxy @ vert_y_pred_arr[:, :, self.pose_timestep]
        vert_x_1 = Pyx @ vert_x_pred_arr[:, :, self.pose_timestep]
        self.compute_alignment_loss(vert_x, vert_y, vert_x_1, vert_y_1)

        # compute smoothness regularisation for point map
        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'].squeeze(0), data_y['L'].squeeze(0)
            self.compute_smoothness_loss(Pxy, Pyx, Lx, Ly, vert_x, vert_y)

        if self.pose_timestep > 0 and 'symmetry_loss' in self.losses:
            # [T+1, n_vert_x, 3]
            shape_x_diff_arr = self.compute_interpolation_difference(vert_x_pred_arr, vert_y_pred_arr, Pxy)
            # [T+1, n_vert_y, 3]
            shape_y_diff_arr = self.compute_interpolation_difference(vert_y_pred_arr, vert_x_pred_arr, Pyx)

            # compute symmetry loss
            self.compute_symmetry_loss(shape_x_diff_arr, shape_y_diff_arr)

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False):
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def compute_displacement(self, vert_x, vert_y, face_x, Pxy=None, p2p_xy=None):
        n_vert_x, n_vert_y = vert_x.shape[0], vert_y.shape[0]

        # construct time step
        step_size = 1 / (self.pose_timestep + 1)
        # [T+1, 1, 1]
        time_steps = step_size + torch.arange(0, 1, step_size,
                                              device=self.device, dtype=torch.float32).unsqueeze(1).unsqueeze(2)

        # [T+1, 1, 7]
        time_steps_up = time_steps * (torch.tensor([0, 0, 0, 0, 0, 0, 1],
                                                   device=self.device, dtype=torch.float32)).unsqueeze(0).unsqueeze(1)
        # [1, n_vert_x, 7]
        vert_y_align = torch.mm(Pxy, vert_y) if p2p_xy is None else vert_y[p2p_xy]
        inputs = torch.cat((
            vert_x, vert_y_align - vert_x,
            torch.zeros(size=(n_vert_x, 1), device=self.device, dtype=torch.float32)
        ), dim=1).unsqueeze(0)
        # [T+1, n_vert_x, 7]
        inputs = inputs + time_steps_up

        # [n_vert_x, 3, T+1]
        displacements = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3), device=self.device, dtype=torch.float32)
        for i in range(inputs.shape[0]):
            displacements[i] = self.networks['interpolator'](inputs[i].unsqueeze(0), face_x.unsqueeze(0)).squeeze(0)

        vert_x_pred_arr = vert_x.unsqueeze(0) + displacements * time_steps
        vert_x_pred_arr = vert_x_pred_arr.permute([1, 2, 0]).contiguous()  # [n_vert_x, 3, T+1]

        return vert_x_pred_arr

    def compute_interpolation_difference(self, vert_x_pred_arr, vert_y_pred_arr, Pxy):
        n_vert_x = vert_x_pred_arr.shape[0]
        shape_x_diff_arr = torch.zeros(self.pose_timestep, n_vert_x, 3, device=self.device, dtype=torch.float32)
        for i in range(self.pose_timestep):
            shape_x_diff_arr[i] = vert_x_pred_arr[:, :, i] - \
                                  torch.mm(Pxy, vert_y_pred_arr[:, :, self.pose_timestep - 1 - i])

        return shape_x_diff_arr

    def compute_fmap_loss(self, data_x, data_y, feat_x, feat_y, Pxy, Pyx):
        # get spectral operators
        evals_x, evals_y = data_x['evals'], data_y['evals']
        evecs_x, evecs_y = data_x['evecs'], data_y['evecs']
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'], data_y['evecs_trans']

        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

        if 'couple_loss' in self.losses:
            Cyx_est, Cxy_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y)), \
                               torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))

            self.loss_metrics['l_couple'] = self.losses['couple_loss'](Cxy, Cxy_est) + \
                                         self.losses['couple_loss'](Cyx, Cyx_est)

    def compute_pose_interpolation_loss(self, vert_x, vert_x_pred_arr, face_x):
        # vert_x_pred_arr [n_vert_x, 3, T+1]
        neigh_x = get_neigh(face_x)
        if 'l_arap' not in self.loss_metrics:
            self.loss_metrics['l_arap'] = 0.0
        self.loss_metrics['l_arap'] += self.losses['arap_loss'](vert_x, vert_x_pred_arr[:, :, 0], neigh_x) + \
                                       self.losses['arap_loss'](vert_x_pred_arr[:, :, 0], vert_x, neigh_x)

        if self.pose_timestep > 0:
            if 'l_arap_interim' not in self.loss_metrics:
                self.loss_metrics['l_arap_interim'] = 0.0
            for tp in range(self.pose_timestep):
                    self.loss_metrics['l_arap_interim'] += self.losses['arap_loss'](vert_x_pred_arr[:, :, tp],
                                                                                    vert_x_pred_arr[:, :, tp + 1],
                                                                                    neigh_x)
                    self.loss_metrics['l_arap_interim'] += self.losses['arap_loss'](vert_x_pred_arr[:, :, tp + 1],
                                                                                    vert_x_pred_arr[:, :, tp],
                                                                                    neigh_x)

    def compute_symmetry_loss(self, shape_x_diff_arr, shape_y_diff_arr):
        self.loss_metrics['l_sym'] = (self.losses['symmetry_loss'](shape_x_diff_arr) +
                                      self.losses['symmetry_loss'](shape_y_diff_arr))

    def compute_alignment_loss(self, vert_x, vert_y, vert_x_1, vert_y_1):
        self.loss_metrics['l_align'] = self.losses['align_loss'](vert_x, vert_y_1) + \
                                       self.losses['align_loss'](vert_y, vert_x_1)

    def compute_smoothness_loss(self, Pxy, Pyx, Lx, Ly, vert_x, vert_y):
        if 'dirichlet_loss' in self.losses:
            Pxy, Pyx = Pxy.unsqueeze(0), Pyx.unsqueeze(0)
            Lx, Ly = Lx.unsqueeze(0), Ly.unsqueeze(0)
            vert_x, vert_y = vert_x.unsqueeze(0), vert_y.unsqueeze(0)
            self.loss_metrics['l_dirichlet'] = (self.losses['dirichlet_loss'](torch.bmm(Pxy, vert_y), Lx) +
                                                self.losses['dirichlet_loss'](torch.bmm(Pyx, vert_x), Ly))

    def optimize_parameters(self):
        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def validate_single(self, data, timer, tb_logger, index):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
        vert_x, face_x = data_x['verts'], data_x['faces']
        vert_y, face_y = data_y['verts'], data_y['faces']

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()

        # start record
        timer.start()

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        if self.non_isometric:
            Pxy, Pyx = None, None
            p2p_yx = nn_query(feat_x.squeeze(0), feat_y.squeeze(0))
            p2p_xy = nn_query(feat_y.squeeze(0), feat_x.squeeze(0))
        else:
            Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
            Pxy, Pyx = Pxy.squeeze(0), Pyx.squeeze(0)
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)
            Cyx = evecs_trans_x @ (Pxy @ evecs_y)
            # convert functional map to point-to-point map
            p2p_yx = fmap2pointmap(Cxy, evecs_x, evecs_y)
            p2p_xy = fmap2pointmap(Cyx, evecs_y, evecs_x)

        if tb_logger is not None or self.pose_alignment:
            # [n_vert_x, 3, T+1]
            vert_x_pred_arr = self.compute_displacement(vert_x.squeeze(0), vert_y.squeeze(0),
                                                        face_x.squeeze(0), Pxy, p2p_xy)
            # [n_vert_y, 3, T+1]
            vert_y_pred_arr = self.compute_displacement(vert_y.squeeze(0), vert_x.squeeze(0),
                                                        face_y.squeeze(0), Pyx, p2p_yx)

            if self.pose_alignment:
                # alignment based on chamfer distance
                mid = self.pose_timestep // 2
                data_x['verts'] = vert_x_pred_arr[:, :, mid]
                data_y['verts'] = vert_y_pred_arr[:, :, self.pose_timestep - mid - 1]
                dx = data_y['verts'][p2p_xy] - data_x['verts']
                vert_x_star = compute_deformation(data_x, data_y, n_iter=self.optim_iter, dx=dx)

                # nearest neighbour query
                p2p_yx = nn_query(vert_x_star, data_y['verts']).squeeze()
        else:
            print('No pose alignment refinement')

        # compute Pyx from functional map
        Cxy = evecs_trans_y @ evecs_x[p2p_yx]
        Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # save the visualization
        if tb_logger is not None and index % 60 == 1:
            step = self.curr_iter // self.opt['val']['val_freq']
            tb_logger.add_mesh(f'{index}/{0}', vertices=vert_x, faces=face_x, global_step=step)

            # add topo transfer and original
            vert_y_align = vert_y[0][p2p_xy]
            tb_logger.add_mesh(f'{index}/{self.pose_timestep + 2}',
                               vertices=vert_y_align.unsqueeze(0),
                               faces=face_x, global_step=step)
            tb_logger.add_mesh(f'{index}/{self.pose_timestep + 3}', vertices=vert_y, faces=face_y,
                               global_step=step)

            for i in range(self.pose_timestep + 1):
                point_pred = vert_x_pred_arr[..., i].unsqueeze(0)
                tb_logger.add_mesh(f'{index}/{i + 1}', vertices=point_pred, faces=face_x,
                                   global_step=step)

        return p2p_yx, Pyx, Cxy

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(SpectralMeetsSpatialModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
