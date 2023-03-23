import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Autoencoder3D(nn.Module):
    def __init__(self, cf: config.Config):
        super(Autoencoder3D, self).__init__()
        self.cf = cf

        self.stage1 = nn.Sequential( # -------------------- output shape: 32^3
            nn.Conv3d(1, 16, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential( # -------------------- output shape: 16^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(  # -------------------- output shape: 8^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(  # ----------------------- output shape: 16^3
            nn.ConvTranspose3d(64, 32, (4, 4, 4), stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(64, 32, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, (4, 4, 4), stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(32, 16, (3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, (3, 3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        """Forward pass (cf. nn.Module)"""
        # x.shape = BS x 32x32x32

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        u1 = self.up1(s3)
        u2 = self.up2(torch.cat((u1,s2),1))
        y = self.out(torch.cat((u2,s1),1))
        return y
