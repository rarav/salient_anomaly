import torch
import torch.nn as nn
from torch.nn import functional
import config
import tools
from matplotlib import pyplot as plt


# function that fits a plane to a set of points
def match_plane(points):
    """Fits a plane to a set of points
    :param points: points.shape = N x 3
    """

    # get the mean of the points (center of mass) shape = 1 x 3
    mean = torch.mean(points, dim=0, keepdim=True)

    # center the points
    centered_points = points - mean

    # compute the covariance matrix
    cov = torch.matmul(centered_points.T, centered_points)

    # compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eig(cov)

    # get the eigenvector corresponding to the largest eigenvalue
    normal = eigenvectors[:, 2].float()

    # compute the distance of the plane to the origin
    d = torch.dot(normal, mean[0])

    return normal, d


def draw_plane_to_voxelgrid(plane, voxelgrid, threshold):
    """Draws a plane to a voxelgrid (sets the voxels to one if they are within a threshold distance to the plane)

    :param plane: plane parameters (normal, d)
    :param voxelgrid: voxelgrid.shape = 1 x 32 x 32 x 32
    """

    # get the coordinates of all
    coords = torch.argwhere(torch.ones_like(voxelgrid)).float()

    # compute the distance of the voxels to the plane

    distances = torch.abs(torch.matmul(coords, plane[0].reshape(3, 1)) - plane[1])[:, 0]

    # set the value of the voxels to one if the distance is smaller than the threshold
    toset = coords[distances < threshold].int()

    voxelgrid[toset[:, 0], toset[:, 1], toset[:, 2]] = 1

    return voxelgrid


class PlaneMatcher(nn.Module):
    def __init__(self):
        super(PlaneMatcher, self).__init__()

    def forward(self, x):
        """Forward pass (cf. nn.Module)
        :param x: input tensor (Voxel grid) x.shape = BS x 32x32x32
        """

        with torch.no_grad():
            x = torch.clip(x, 0, 1)
            for i in range(x.shape[0]):
                x[i] = draw_plane_to_voxelgrid(match_plane(torch.argwhere(x[i]).float()), x[i], 0.5)
            x = torch.unsqueeze(x, 1)

        return x


def test_plane_match():
    """Test function for the plane matching module"""

    # create a voxelgrid
    voxelgrid = torch.zeros(1, 24, 24, 24)

    # set the value of a few voxels to one
    voxelgrid[0, 14, 2, 15] = 10
    voxelgrid[0, 0, 0, 5] = 10
    voxelgrid[0, 2, 23, 18] = 10
    voxelgrid[0, 20, 20, 20] = 10

    # create a plane matching module
    plane_matcher = PlaneMatcher()

    # apply the plane matching module to the voxelgrid
    voxelgrid = plane_matcher(voxelgrid)

    # visualise the voxelgrid
    tools.visualize_dense_grid(voxelgrid[0, 0].detach().cpu().numpy(), 'test_plane_match')
    plt.show()


if __name__ == '__main__':
    test_plane_match()


# ---------------- U-Net like Network

class Autoencoder3D(nn.Module):
    def __init__(self, cf: config.Config, device):
        super(Autoencoder3D, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        print(f1)
        f2 = f1 * 2
        f4 = f1 * 4
        self.shift = (cf.DATA.IN_SIZE ** 2 - (cf.DATA.IN_SIZE - 2 * cf.DATA.SHELL_SIZE) ** 2) / cf.DATA.IN_SIZE ** 3
        print('Shift:', self.shift)

        pm = 'zeros'

        self.b1 = nn.Sequential(  # -------------------------------------- shape: 32^3 / 24^3
            nn.Conv3d(1, f1, (3, 3, 3), padding=1, padding_mode='zeros'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, f1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )
        self.b2 = nn.Sequential(  # -------------------------------------- shape: 16^3 / 12^3
            nn.Conv3d(f1, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f2, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )
        self.b3 = nn.Sequential(  # -------------------------------------- shape: 8^3 / 6^3
            nn.Conv3d(f2, f4, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )

        self.b4 = nn.Sequential(  # -------------------------------------- shape: 16^3 / 12^3
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )

        self.b5 = nn.Sequential(  # -------------------------------------- shape: 32^3 / 24^3
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, 1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.Sigmoid(),
        )
        self.to(device)

        # s = cf.DATA.SHELL_SIZE
        # self.shell_mask = torch.ones(
        #     (cf.TRAIN.BTSZ, 1, cf.DATA.IN_SIZE, cf.DATA.IN_SIZE, cf.DATA.IN_SIZE), device=device)
        # self.shell_mask[:, :, s:-s - 1, s:-s - 1, s:-s - 1] = 0
        # self.shell_mask = self.shell_mask - 0.5
        # self.shell_mask.requires_grad = False

    def forward(self, x):
        """Forward pass (cf. nn.Module)
        x.shape = BS x 32x32x32
        """

        x = torch.clip(x, 0, 1)
        x = torch.unsqueeze(x, 1) - self.shift

        # x = torch.cat((x, self.shell_mask), dim=1)

        x1 = self.b1(x)
        x2 = self.b2(functional.interpolate(x1, scale_factor=0.5))
        x3 = self.b3(functional.interpolate(x2, scale_factor=0.5))
        # x2 = self.b2(functional.avg_pool3d(x1, 2, 2))
        # x3 = self.b3(functional.avg_pool3d(x2, 2, 2))

        x4 = self.b4(torch.cat((x2, functional.interpolate(x3, scale_factor=2)), dim=1))
        x5 = self.b5(torch.cat((x1, functional.interpolate(x4, scale_factor=2)), dim=1))
        return x5


# ---------------- Residual tower-of-power

class ResBlock3D(nn.Module):
    def __init__(self, channels):
        """Resblock increases receptive field by +8 px"""
        super(ResBlock3D, self).__init__()
        self.c3 = nn.Conv3d(channels, channels, (3, 3, 3), padding=1, padding_mode='zeros', groups=channels)
        self.c5 = nn.Conv3d(channels, channels, (3, 3, 3), dilation=2, padding=2, padding_mode='zeros', groups=channels)
        self.c7 = nn.Conv3d(channels, channels, (3, 3, 3), dilation=3, padding=3, padding_mode='zeros', groups=channels)

        self.cm = nn.Conv3d(3 * channels, channels, (3, 3, 3), padding=1)

    def forward(self, x):
        c1 = torch.cat((self.c3(x), self.c5(x), self.c7(x)), dim=1)
        c2 = self.cm(functional.leaky_relu(c1))
        return x + functional.leaky_relu(c2)


class Autoencoder3DRes(nn.Module):
    def __init__(self, cf: config.Config, device):
        super(Autoencoder3DRes, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        # RF = 5+2+4*8 = 37

        self.b1 = nn.Sequential(  # -------------------------------------- shape: 32^3
            nn.Conv3d(1, f1 * 2, (3, 3, 3), padding=1, padding_mode='zeros'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1 * 2, f1, (3, 3, 3), padding=1, padding_mode='zeros'),
            ResBlock3D(f1),
            ResBlock3D(f1),
            ResBlock3D(f1),
            ResBlock3D(f1),
            nn.Conv3d(f1, 1, (1, 1, 1)),
            nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, x):
        """Forward pass (cf. nn.Module)

        The input values are encoded as (number of points - 1).
        Input is clipped: 0/1 point -> 0 and >1 points -> 1
        :param x: input grid shell, shape = BSx32x32x32
        """

        x = torch.clip(x, 0, 1)
        x = torch.unsqueeze(x, 1) - 0.03

        return self.b1(x)
