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
    where = np.zeros(num_p,np.bool_)
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

        #pcl = o3d.geometry.PointCloud()
        #pcl.points = o3d.utility.Vector3dVector(points)  # N,3
        #pcls.append(pcl)

        names.append(name)

        to_load -= 1
        if to_load == 0: break

    return pcls, names


class TrainingDataset(Dataset):
    def __init__(self, cf: config.Config):
        side_length = cf.DATA.IN_SIZE
        self.voxel_size = cf.DATA.VOXEL_SIZE
        self.crop_size = cf.DATA.IN_SIZE * cf.DATA.VOXEL_SIZE
        self.dense_grid = np.zeros((side_length, side_length, side_length))
        self.shell_grid = np.zeros((side_length, side_length, side_length))
        self.shell_size = cf.DATA.SHELL_SIZE

        pcls, names = preload(cf.PATHS.TRAIN)
        self.pcls = pcls
        self.names = names

        self.ds_size = cf.TRAIN.IT_P_EP * cf.TRAIN.BTSZ

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        return self.get_random_voxel_crop()

    def get_random_voxel_crop(self):
        pcl = random.choice(self.pcls)

        # pcd.rotate()
        num_points = pcl.shape[0]
        center = pcl[random.randrange(0, num_points),:]  # 3,
        min_bound = center - self.crop_size / 2.0  # 3,
        max_bound = min_bound + self.crop_size - self.voxel_size / 2  # 3,

        crop = crop_pcl(pcl,min_bound,max_bound)
        voxelise(self.dense_grid, crop, min_bound, self.voxel_size)

        create_shell(self.shell_grid, self.dense_grid, self.shell_size)
        return {'dense': torch.from_numpy(self.dense_grid).float(),
                'shell': torch.from_numpy(self.shell_grid).float()}
