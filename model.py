import torch
import torch.nn as nn
import config

class Autoencoder3D(nn.Module):
    def __init__(self, cf: config.Config):
        super(Autoencoder3D, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        f2 = f1*2
        f4 = f1*4
        f8 = f1*8

        self.stage1 = nn.Sequential(  # -------------------- output shape: 32^3
            nn.BatchNorm3d(1),
            nn.Conv3d(1, f1, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f1, f1, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(  # -------------------- output shape: 16^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(f1),
            nn.Conv3d(f1, f2, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f2, f2, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(  # -------------------- output shape: 8^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(f2),
            nn.Conv3d(f2, f4, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(  # ----------------------- output shape: 16^3
            nn.ConvTranspose3d(f4, f2, (4, 4, 4), stride=2, padding=1,bias=False),
            nn.BatchNorm3d(f2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(f2, f1, (4, 4, 4), stride=2, padding=1,bias=False),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1,bias=False),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f1, 1, (3, 3, 3), padding=1,bias=False),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        # x.shape = BS x 32x32x32
        x = torch.unsqueeze(x, 1)

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        u1 = self.up1(s3)
        u2 = self.up2(torch.cat((u1, s2), 1))
        y = self.out(torch.cat((u2, s1), 1))
        return y
