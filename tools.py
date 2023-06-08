import numpy as np
import datetime, os
import torch
import matplotlib.pyplot as plt


def visualize_dense_grid(dense_grid, name='default', colorize=True):
    colors = np.empty(dense_grid.shape, dtype=object)

    if colorize:
        colors[dense_grid <= 3] = 'lightsteelblue'
        colors[np.logical_and(dense_grid > 3, dense_grid <= 6)] = 'cornflowerblue'
        colors[np.logical_and(dense_grid > 6, dense_grid <= 10)] = 'royalblue'
        colors[np.logical_and(dense_grid > 10, dense_grid <= 15)] = 'grey'
        colors[np.logical_and(dense_grid > 15, dense_grid <= 20)] = 'rosybrown'
        colors[np.logical_and(dense_grid > 20, dense_grid <= 30)] = 'darkred'
        colors[dense_grid > 30] = 'red'

    else:
        colors[dense_grid > 0] = 'lightsteelblue'

    ax = plt.figure(name).add_subplot(projection='3d')
    ax.voxels(dense_grid > 0, facecolors=colors, shade=False, edgecolor='black')
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
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


def rotate(points, yaw, pitch, roll):
    """
    Rotates points (nx3) by Euler angles in the order of z-y'-x" (pitch-roll-yaw)

    points: nx3 numpy array
    yaw: angle around z-axis, degrees
    pitch: angle around x-axis, degrees
    roll: angle around y-axis, degrees
    """
    omega = np.radians(pitch)
    phi = np.radians(roll)
    kappa = np.radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(omega), np.sin(omega)],
                    [0, -np.sin(omega), np.cos(omega)]])

    R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])

    R_z = np.array([[np.cos(kappa), np.sin(kappa), 0],
                    [-np.sin(kappa), np.cos(kappa), 0],
                    [0, 0, 1]])
    R = R_x.dot(R_y.dot(R_z))

    # rotate points
    r_pts = R.dot(points.T)
    return r_pts.T
