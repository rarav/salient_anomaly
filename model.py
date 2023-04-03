import torch
import torch.nn as nn
from torch.nn import functional
import config


class Autoencoder3D(nn.Module):
    def __init__(self, cf: config.Config, device):
        super(Autoencoder3D, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        f2 = f1 * 2
        f4 = f1 * 4
        f8 = f1 * 8

        pm = 'replicate'

        self.b1 = nn.Sequential(  # -------------------------------------- shape: 32^3
            nn.Conv3d(1, f1, (5, 5, 5), padding=2, padding_mode='zeros'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, f1, (5, 5, 5), padding=2, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )
        self.b2 = nn.Sequential(  # -------------------------------------- shape: 16^3
            # nn.InstanceNorm3d(f1, track_running_stats=False),
            nn.Conv3d(f1, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f2, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )
        self.b3 = nn.Sequential(  # -------------------------------------- shape: 8^3
            # nn.InstanceNorm3d(f2, track_running_stats=False),
            nn.Conv3d(f2, f4, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )

        self.b4 = nn.Sequential(  # -------------------------------------- shape: 16^3
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
        )

        self.b5 = nn.Sequential(  # -------------------------------------- shape: 32^3
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, 1, (3, 3, 3), padding=1, padding_mode=pm),
            nn.Sigmoid(),
        )

        s = cf.DATA.SHELL_SIZE
        self.shell_mask = torch.ones(
            (cf.TRAIN.BTSZ, 1, cf.DATA.IN_SIZE, cf.DATA.IN_SIZE, cf.DATA.IN_SIZE), device=device)
        self.shell_mask[:, :, s:-s - 1, s:-s - 1, s:-s - 1] = 0
        self.shell_mask = self.shell_mask - 0.5
        self.shell_mask.requires_grad = False
        self.to(device)

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        # x.shape = BS x 32x32x32

        x = torch.unsqueeze(x, 1) - 0.03
        # x = torch.cat((x, self.shell_mask), dim=1)

        x1 = self.b1(x)
        x2 = self.b2(functional.avg_pool3d(x1, 2, 2))
        x3 = self.b3(functional.avg_pool3d(x2, 2, 2))

        x4 = self.b4(torch.cat((x2, functional.interpolate(x3, scale_factor=2)), dim=1))
        x5 = self.b5(torch.cat((x1, functional.interpolate(x4, scale_factor=2)), dim=1))
        return x5


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()
        self.c3 = nn.Conv3d(channels, channels, (3, 3, 3), padding=1, padding_mode='zeros', groups=channels)
        self.c5 = nn.Conv3d(channels, channels, (3, 3, 3), dilation=2, padding=2, padding_mode='zeros', groups=channels)
        self.c7 = nn.Conv3d(channels, channels, (3, 3, 3), dilation=3, padding=3, padding_mode='zeros', groups=channels)

        self.cm = nn.Conv3d(3 * channels, channels, (1, 1, 1))

    def forward(self, x):
        c1 = torch.cat((self.c3(x), self.c5(x), self.c7(x)), dim=1)
        c2 = self.cm(functional.leaky_relu(c1))
        return x + functional.leaky_relu(c2)


class Autoencoder3DRes(nn.Module):
    def __init__(self, cf: config.Config, device):
        super(Autoencoder3DRes, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        f2 = f1 * 2
        f4 = f1 * 4

        self.b1 = nn.Sequential(  # -------------------------------------- shape: 32^3
            nn.Conv3d(1, f4, (5, 5, 5), padding=2, padding_mode='zeros'),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f2, (1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            ResBlock3D(f2),
            ResBlock3D(f2),
            ResBlock3D(f2),
            ResBlock3D(f2),
            nn.Conv3d(f2, 1, (1, 1, 1)),
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
