import numpy as np
import datetime, os
import torch
import matplotlib.pyplot as plt


def visualize_dense_grid(dense_grid, name='default'):
    colors = np.empty(dense_grid.shape, dtype=object)
    colors[dense_grid <= 3] = 'lightsteelblue'
    colors[np.logical_and(dense_grid > 3, dense_grid <= 6)] = 'cornflowerblue'
    colors[np.logical_and(dense_grid > 6, dense_grid <= 10)] = 'royalblue'
    colors[np.logical_and(dense_grid > 10, dense_grid <= 15)] = 'grey'
    colors[np.logical_and(dense_grid > 15, dense_grid <= 20)] = 'rosybrown'
    colors[np.logical_and(dense_grid > 20, dense_grid <= 30)] = 'darkred'
    colors[dense_grid > 30] = 'red'

    ax = plt.figure(name).add_subplot(projection='3d')
    ax.voxels(dense_grid > 0, facecolors=colors, shade=False, edgecolor=None)
    #plt.show()


def sek2hms(sec: int):
    """Convert seconds to hour, min, seconds"""

    min = sec // 60
    sec -= min * 60
    hr = min // 60
    min -= hr * 60
    return hr, min, sec

def current_datetime_as_str() -> str:
    """Return a string that represents the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def tensor2numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a np.ndarray"""
    return t.cpu().data.numpy()


class FolderDict:
    data: dict

    def __init__(self, data):
        """This class stores a dict where the values are paths.

        The path is only created if the item is accessed by __getitem__.
        :param data: The dictionary containing folder paths
        """
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        os.makedirs(path, exist_ok=True)
        return path

