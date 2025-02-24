import os, re
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob

import torch
from torch.utils.data import Dataset

from utils.shape_util import read_shape
from utils.geometry_util import get_operators
from utils.registry import DATASET_REGISTRY


def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)

    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L.to_dense()


    return item


class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_evecs=True, num_evecs=120,
                 return_corr=True, return_dist=False):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs

        self.off_files = []
        self.corr_files = [] if self.return_corr else None
        self.dist_files = [] if self.return_dist else None

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        if self.return_dist:
            assert self._size == len(self.dist_files)

        if self.return_corr:
            assert self._size == len(self.corr_files)

    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))

    def __getitem__(self, index):
        item = dict()

        # get vertices
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename
        item['index'] = index
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs,
                                    cache_dir=os.path.join(self.data_root, 'diffusion'))

        # get geodesic distance matrix
        if self.return_dist:
            mat = sio.loadmat(self.dist_files[index])
            item['dist'] = torch.from_numpy(mat['dist']).float()

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['corr'] = torch.from_numpy(corr).long()

        return item

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=120,
                 return_corr=True, return_dist=False):
        super(SingleFaustDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs,
                                                 return_corr, return_dist)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            self._size = 80
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            self._size = 20


class PairShapeDataset(Dataset):
    def __init__(self, dataset, num_shapes=None):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = list(product(range(len(dataset)), repeat=2))
        self.num_shapes = num_shapes

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        if self.num_shapes is not None:
            return self.num_shapes
        else:
            return len(self.combinations)


@DATASET_REGISTRY.register()
class PairDataset(PairShapeDataset):
    def __init__(self, data_root, return_faces=True,
                 return_evecs=True, num_evecs=120,
                 return_corr=True, return_dist=False):
        dataset = SingleShapeDataset(data_root, return_faces, return_evecs, num_evecs,
                                     return_corr, return_dist)
        super(PairDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class OnePairDataset(Dataset):
    def __init__(self, first_shape, second_shape, num_evecs):
        assert os.path.isfile(first_shape) and os.path.isfile(second_shape)
        self.first_shape = first_shape
        self.second_shape = second_shape
        self.num_evecs = num_evecs

    def get_shape(self, data_path):
        item = dict()

        # get vertices
        verts, faces = read_shape(data_path)
        item['verts'] = torch.from_numpy(verts).float()
        item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        item = get_spectral_ops(item, num_evecs=self.num_evecs)

        return item

    def __len__(self):
        return 1

    def __getitem__(self, index):
        item = dict()
        item['first'] = self.get_shape(self.first_shape)
        item['second'] = self.get_shape(self.second_shape)

        return item


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=120,
                 return_corr=True, return_dist=False):
        dataset = SingleFaustDataset(data_root, phase, return_faces,
                                     return_evecs, num_evecs,
                                     return_corr, return_dist)
        super(PairFaustDataset, self).__init__(dataset)
