import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.diffusion_network import DiffusionNet
from networks.edgeconv_network import ResnetECPos
from networks.permutation_network import Similarity
from datasets.shape_dataset import OnePairDataset
from utils.tensor_util import to_device, to_numpy
from utils.shape_util import write_off
from utils.fmap_util import fmap2pointmap, nn_query
from models.sms_model import compute_deformation
from metrics.geodist_metric import calculate_geodesic_error


@torch.no_grad()
def compute_permutation_matrix(feat_x, feat_y, permutation):
    feat_x = F.normalize(feat_x, dim=-1, p=2)
    feat_y = F.normalize(feat_y, dim=-1, p=2)
    similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

    Pxy = permutation(similarity)
    Pyx = permutation(similarity.transpose(1, 2))

    return Pxy, Pyx


@torch.no_grad()
def compute_displacement(vert_x, vert_y, face_x, p2p_xy, interpolator, pose_timestep, device='cuda'):
    n_vert_x, n_vert_y = vert_x.shape[0], vert_y.shape[0]

    # construct time step
    step_size = 1 / (pose_timestep + 1)
    # [T+1, 1, 1]
    time_steps = step_size + torch.arange(0, 1, step_size,
                                            device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(2)

    # [T+1, 1, 7]
    time_steps_up = time_steps * (torch.tensor([0, 0, 0, 0, 0, 0, 1],
                                                device=device, dtype=torch.float32)).unsqueeze(0).unsqueeze(1)

    # [1, n_vert_x, 7]
    vert_y_align = vert_y[p2p_xy]
    inputs = torch.cat((
        vert_x, vert_y_align - vert_x,
        torch.zeros(size=(n_vert_x, 1), device=device, dtype=torch.float32)
    ), dim=1).unsqueeze(0)
    # [T+1, n_vert_x, 7]
    inputs = inputs + time_steps_up

    # [n_vert_x, 3, Tp]
    displacements = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3), device=device, dtype=torch.float32)
    for i in range(inputs.shape[0]):
        displacements[i] = interpolator(inputs[i].unsqueeze(0), face_x.unsqueeze(0)).squeeze(0)

    vert_x_pred_arr = vert_x.unsqueeze(0) + displacements * time_steps
    vert_x_pred_arr = vert_x_pred_arr.permute([1, 2, 0]).contiguous()  # [n_vert_x, 3, T+1]

    return vert_x_pred_arr


if __name__ == '__main__':
    # FAUST
    data_root = '../data/FAUST_aligned/'
    prefix = 'tr_reg_'
    network_path = 'checkpoints/faust.pth'
    result_root = 'results/faust'
    first_iter_range = range(80, 90)
    iter_range = range(90, 100)
    n_iter = 1000
    non_isometric = False

    # specify pose step and shape step
    pose_timestep = 7
    shape_timestep = 3

    os.makedirs(result_root, exist_ok=True)

    # initialize networks
    feature_extractor = DiffusionNet(in_channels=128, out_channels=384,
                                     input_type='wks', cache_dir=os.path.join(data_root, 'diffusion')).cuda()
    permutation = Similarity(tau=0.07, hard=True).cuda()
    interpolator = ResnetECPos(c_dim=3, dim=7, hidden_dim=128, use_mlp=False).cuda()

    state_dict = torch.load(network_path)
    feature_extractor.load_state_dict(state_dict['networks']['feature_extractor'])
    interpolator.load_state_dict(state_dict['networks']['interpolator'])
    feature_extractor.eval()
    interpolator.eval()
    print('Load pretrained networks')

    num_evecs = 300
    fisrt_pbar = tqdm(first_iter_range, leave=False)
    pbar = tqdm(iter_range)
    for j in fisrt_pbar:
        for i in pbar:
            first_shape_index = f'{j:03d}'
            second_shape_index = f'{i:03d}'
            first_shape = os.path.join(data_root, 'off', f'{prefix}{first_shape_index}.off')
            second_shape = os.path.join(data_root, 'off', f'{prefix}{second_shape_index}.off')

            dataset = OnePairDataset(first_shape, second_shape, num_evecs=num_evecs)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # get data pair
            data = next(iter(dataloader))
            data_x, data_y = to_device(data['first'], 'cuda'), to_device(data['second'], 'cuda')
            assert data_x['verts'].shape[0] == 1, 'Only supports batch size = 1.'
            evecs_x = data_x['evecs'].squeeze()
            evecs_y = data_y['evecs'].squeeze()
            evecs_trans_x = data_x['evecs_trans'].squeeze()
            evecs_trans_y = data_y['evecs_trans'].squeeze()
            Lx, Ly = data_x['L'].squeeze(), data_x['L'].squeeze()

            with torch.no_grad():
                # extract feature
                feat_x = feature_extractor(data_x['verts'], data_x['faces'])  # [B, Nx, C]
                feat_y = feature_extractor(data_y['verts'], data_y['faces'])  # [B, Ny, C]

                # compute permutation matrices
                Pxy, Pyx = compute_permutation_matrix(feat_x, feat_y, permutation)  # [B, Nx, Ny], [B, Ny, Nx]

            if non_isometric:
                p2p_yx = nn_query(feat_x.squeeze(0), feat_y.squeeze(0))
                p2p_xy = nn_query(feat_y.squeeze(0), feat_x.squeeze(0))
            else:
                Pxy, Pyx = Pxy.squeeze(0), Pyx.squeeze(0)
                Cxy = evecs_trans_y @ (Pyx @ evecs_x)
                Cyx = evecs_trans_x @ (Pxy @ evecs_y)
                # convert functional map to point-to-point map
                p2p_yx = fmap2pointmap(Cxy, evecs_x, evecs_y)
                p2p_xy = fmap2pointmap(Cyx, evecs_y, evecs_x)

            vert_x, vert_y = data_x['verts'].squeeze(0), data_y['verts'].squeeze(0)
            face_x, face_y = data_x['faces'].squeeze(0), data_y['faces'].squeeze(0)

            # from shape x to shape y [n_vert_x, 3, Tp]
            vert_x_pred_arr = compute_displacement(vert_x, vert_y, face_x, p2p_xy, interpolator, pose_timestep).squeeze(-1)

            # from shape y to shape x
            vert_y_pred_arr = compute_displacement(vert_y, vert_x, face_y, p2p_yx, interpolator, pose_timestep).squeeze(-1)

            # compute geodesic error
            # get geodesic distance matrix
            dist_x = sio.loadmat(os.path.join(data_root, 'dist', f'{prefix}{first_shape_index}.mat'))['dist']

            # get gt correspondence
            corr_x = np.loadtxt(os.path.join(data_root, 'corres', f'{prefix}{first_shape_index}.vts'), dtype=np.int32) - 1
            corr_y = np.loadtxt(os.path.join(data_root, 'corres', f'{prefix}{second_shape_index}.vts'), dtype=np.int32) - 1

            # compute geodesic error
            geo_err = calculate_geodesic_error(dist_x, corr_x, corr_y, to_numpy(p2p_yx), return_mean=True)

            # align
            vert_x_align = vert_x[p2p_yx]
            vert_y_align = vert_y[p2p_xy]

            # save results
            save_path = f'{result_root}/{first_shape_index.replace(".off", "")}-{second_shape_index.replace(".off", "")}/'
            os.makedirs(save_path, exist_ok=True)

            face_x, face_y = to_numpy(face_x), to_numpy(face_y)
            vert_x_align, vert_y_align = to_numpy(vert_x_align), to_numpy(vert_y_align)
            write_off(os.path.join(save_path, 'shape_x.off'), to_numpy(vert_x), face_x)
            write_off(os.path.join(save_path, 'shape_y.off'), to_numpy(vert_y), face_y)
            write_off(os.path.join(save_path, 'shape_align_x.off'), vert_x_align, face_y)
            write_off(os.path.join(save_path, 'shape_align_y.off'), vert_y_align, face_x)

            for i in range(vert_x_pred_arr.shape[-1]):
                vert_x_pred = vert_x_pred_arr[..., i]
                vert_y_pred = vert_y_pred_arr[..., pose_timestep-i-1]

                ind = i
                write_off(os.path.join(save_path, f'shape_{ind}_x.off'), to_numpy(vert_x_pred), face_x)
                write_off(os.path.join(save_path, f'shape_{ind}_y.off'), to_numpy(vert_y_pred), face_y)

                # compute the alignment between vert_x_pred and vert_y_pred
                vert_x_star = vert_y_pred[p2p_xy]


                if shape_timestep > 0:
                    # alignment based on chamfer distance
                    data_x['verts'] = vert_x_pred
                    data_y['verts'] = vert_y_pred
                    vert_x_star = compute_deformation(data_x, data_y, n_iter)
                    # compute the linear interpolation between vert_x_pred and vert_x_star
                    T = shape_timestep
                    for t in range(1, T+1):
                        vert_x_linear = (1 - t/T) * vert_x_pred + t/T * vert_x_star
                        write_off(os.path.join(save_path, f'shape_{ind}_{t}_x.off'), to_numpy(vert_x_linear), face_x)

            # alignment based on chamfer distance
            data_x['verts'] = vert_x_pred_arr[..., -1]
            data_y['verts'] = vert_y
            vert_x_star = compute_deformation(data_x, data_y, n_iter)

            # compute geodesic error based on nearest neighbour search of positions
            p2p_yx = nn_query(vert_x_star, vert_y)
            geo_err_nn = calculate_geodesic_error(dist_x, corr_x, corr_y, to_numpy(p2p_yx), return_mean=True)
            pbar.set_description(f'geo error of {first_shape_index}-{second_shape_index} based on nn: {geo_err_nn:.4f}, fmap {geo_err:.4f}')

            write_off(os.path.join(save_path, f'shape_x_star.off'), to_numpy(vert_x_star), face_x)
