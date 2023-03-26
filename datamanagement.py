import os
from os.path import join as pjoin
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import laspy as lp
from numba import jit
import random
import config


def file_table(root: str) -> pd.DataFrame:
    """Load paths to laz files.

    :param root: Root folder of laz files
    :return: A pandas Dataframe with columns ['Name', 'Path'].
    """
    data = []
    for filename in os.listdir(root):
        if not filename.endswith('laz'): continue

        p_laz = pjoin(root, filename)
        data.append([filename, p_laz])

    return pd.DataFrame(data, columns=['Name', 'Path'])


def create_shell(out, dense_grid, t):
    out[:, :, :] = dense_grid[:, :, :]
    out[t:-t - 1, t:-t - 1, t:-t - 1] = 0


@jit(nopython=True)
def crop_pcl(points, min_bound, max_bound):
    num_p = points.shape[0]
    where = np.zeros(num_p, np.bool_)
    for i in range(num_p):
        x, y, z = points[i]
        if (x < min_bound[0] or x > max_bound[0] or
                y < min_bound[1] or y > max_bound[1] or
                z < min_bound[2] or z > max_bound[2]):
            continue
        where[i] = True
    return points[where]


@jit(nopython=True)
def voxelise(out, points, min_bound, voxel_size):
    out *= 0
    indices = np.floor((points - min_bound) / voxel_size)
    for index in indices:
        out[int(index[0]), int(index[1]), int(index[2])] += 1


def preload(root: str) -> tuple:
    """ Pre-load the data.

    :param root: Root to the laz files.
    :return: tuple of point clouds and names, each being a list.
    """
    print(f'Pre-loading point clouds from {root}')
    table = file_table(root)
    pcls, names = [], []
    to_load = 3
    for _, row in table.iterrows():
        name, path = row
        pcl_las = lp.read(path)
        points = pcl_las.xyz - pcl_las.header.offsets.reshape(1, 3)  # N,3
        print(name, pcl_las)

        pcls.append(points)

        # pcl = o3d.geometry.PointCloud()
        # pcl.points = o3d.utility.Vector3dVector(points)  # N,3
        # pcls.append(pcl)

        names.append(name)

        to_load -= 1
        if to_load == 0: break

    return pcls, names


class TrainingDataset(Dataset):
    def __init__(self, cf: config.Config):
        side_length = cf.DATA.IN_SIZE
        self.voxel_size = cf.DATA.VOXEL_SIZE
        self.crop_size = cf.DATA.IN_SIZE * cf.DATA.VOXEL_SIZE
        self.pre_crop_size = self.crop_size * 2
        self.dense_grid = np.zeros((side_length, side_length, side_length))
        self.shell_grid = np.zeros((side_length, side_length, side_length))
        self.shell_size = cf.DATA.SHELL_SIZE
        self.random_rotation = cf.AUG.ROTATE
        self.random_flip = cf.AUG.FLIP

        pcls, names = preload(cf.PATHS.TRAIN)
        self.pcls = pcls
        self.names = names

        self.ds_size = cf.TRAIN.IT_P_EP * cf.TRAIN.BTSZ
        self.R_z = np.eye(3, dtype=np.float)
        self.zero3 = np.zeros(3, dtype=np.float)

    def rand_rotate(self, pcl):
        kappa = np.random.uniform(-np.pi, np.pi)
        cos_k = np.cos(kappa)
        sin_k = np.sin(kappa)
        self.R_z[0, 0] = cos_k
        self.R_z[1, 1] = cos_k
        self.R_z[0, 1] = sin_k
        self.R_z[1, 0] = -sin_k
        return self.R_z.dot(pcl.T).T

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        return self.get_random_voxel_crop()

    def get_random_voxel_crop(self):
        pcl = random.choice(self.pcls)

        # pcd.rotate()
        num_points = pcl.shape[0]
        center = pcl[random.randrange(0, num_points), :]  # 3,

        # pre - crop
        if self.random_rotation:

            min_bound = center - self.pre_crop_size / 2.0
            max_bound = min_bound + self.pre_crop_size

            pre_crop = crop_pcl(pcl, min_bound, max_bound) - center
            rot_pcl = self.rand_rotate(pre_crop)

            min_bound = self.zero3 - self.crop_size / 2.0
            max_bound = min_bound + self.crop_size - self.voxel_size / 2

            crop = crop_pcl(rot_pcl, min_bound, max_bound)
        else:
            min_bound = center - self.crop_size / 2.0
            max_bound = min_bound + self.crop_size - self.voxel_size / 2

            crop = crop_pcl(pcl, min_bound, max_bound)

        voxelise(self.dense_grid, crop, min_bound, self.voxel_size)

        if self.random_flip:
            if np.random.random(1) > 0.5:
                self.dense_grid[:, :, :] = self.dense_grid[::-1, ::-1, :]

        create_shell(self.shell_grid, self.dense_grid, self.shell_size)
        return {'dense': torch.from_numpy(self.dense_grid).float(),
                'shell': torch.from_numpy(self.shell_grid).float()}


class EvalDataset(Dataset):
    def __init__(self, cf: config.Config, set):
        side_length = cf.DATA.IN_SIZE
        voxel_size = cf.DATA.VOXEL_SIZE
        crop_size = cf.DATA.IN_SIZE * cf.DATA.VOXEL_SIZE
        shell_size = cf.DATA.SHELL_SIZE

        if set == 'validation':
            pcls, names = preload(cf.PATHS.VALIDATION)
        else:
            pcls, names = preload(cf.PATHS.TEST)

        self.ds_size = cf.TEST.NUM_POINTS
        num_clouds = len(pcls)

        self.voxel_grids = []
        self.shell_grids = []
        self.coordinates = []
        self.cloud_names = []

        np.random.seed(0)
        for i in range(self.ds_size):
            r = random.randrange(0, num_clouds)
            pcl = pcls[r]
            num_points = pcl.shape[0]
            center = pcl[random.randrange(0, num_points), :]  # 3,

            min_bound = center - crop_size / 2.0
            max_bound = min_bound + crop_size - voxel_size / 2
            crop = crop_pcl(pcl, min_bound, max_bound)

            dense_grid = np.zeros((side_length, side_length, side_length))
            shell_grid = np.zeros((side_length, side_length, side_length))
            voxelise(dense_grid, crop, min_bound, voxel_size)
            create_shell(shell_grid, dense_grid, shell_size)

            self.voxel_grids.append(dense_grid)
            self.shell_grids.append(shell_grid)
            self.coordinates.append(center)
            self.cloud_names.append(names[i])

    def __len__(self):
        return self.ds_size

    def __getitem__(self, idx):
        return {'dense': torch.from_numpy(self.voxel_grids[idx]).float(),
                'shell': torch.from_numpy(self.shell_grids[idx]).float(),
                'coord': self.coordinates[idx],
                'cname': self.cloud_names[idx]
                }

