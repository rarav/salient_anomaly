import torch, math
import torch.nn as nn
import config


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

# from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.input_conv.apply(weights_init('kaiming'))

        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters(): # mask is not updated
            param.requires_grad = False

    def forward(self, input):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        mask = torch.clip(torch.sign(input),0,1,).detach()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        #new_mask = torch.ones_like(output)
        #new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output#, new_mask


class Autoencoder3DPC(nn.Module):
    def __init__(self, cf: config.Config):
        super(Autoencoder3DPC, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        f2 = f1 * 2
        f4 = f1 * 4
        f8 = f1 * 8

        self.stage1 = nn.Sequential(  # -------------------- output shape: 32^3
            #nn.BatchNorm3d(1),
            PartialConv(1, f1, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            PartialConv(f1, f1, (3, 3, 3), padding=1),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(  # -------------------- output shape: 16^3
            nn.MaxPool3d(2, 2),
            PartialConv(f1, f2, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            PartialConv(f2, f2, (3, 3, 3), padding=1),
            nn.BatchNorm3d(f2),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(  # -------------------- output shape: 8^3
            nn.MaxPool3d(2, 2),
            PartialConv(f2, f4, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            PartialConv(f4, f4, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            PartialConv(f4, f4, (3, 3, 3), padding=1),
            nn.BatchNorm3d(f4),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(  # ----------------------- output shape: 16^3
            nn.ConvTranspose3d(f4, f2, (4, 4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            PartialConv(f2, f2, (3, 3, 3), padding=1),
            nn.BatchNorm3d(f2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(  # ----------------------- output shape: 32^3
            PartialConv(f4, f2, (3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(f2, f1, (4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(  # ----------------------- output shape: 32^3
            PartialConv(f2, f1, (3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True),
            PartialConv(f1, 1, (3, 3, 3), padding=1, bias=False),
            # nn.Sigmoid(),
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

class Autoencoder3D(nn.Module):
    def __init__(self, cf: config.Config):
        super(Autoencoder3D, self).__init__()

        f1 = cf.VAE_MODEL.F_START
        f2 = f1 * 2
        f4 = f1 * 4
        f8 = f1 * 8

        self.stage1 = nn.Sequential(  # -------------------- output shape: 32^3
            nn.Conv3d(1, f1, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, f1, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(  # -------------------- output shape: 16^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(f1),
            nn.Conv3d(f1, f2, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f2, f2, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(  # -------------------- output shape: 8^3
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(f2),
            nn.Conv3d(f2, f4, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f4, f4, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.up1 = nn.Sequential(  # ----------------------- output shape: 16^3
            nn.ConvTranspose3d(f4, f2, (4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(f2),
            nn.LeakyReLU(inplace=True),
        )

        self.up2 = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(f4, f2, (3, 3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(f2, f1, (4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(f1),
            nn.LeakyReLU(inplace=True),
        )

        self.out = nn.Sequential(  # ----------------------- output shape: 32^3
            nn.Conv3d(f2, f1, (3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(f1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(f1, 1, (3, 3, 3), padding=1, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass (cf. nn.Module)"""
        # x.shape = BS x 32x32x32
        x = torch.unsqueeze(x, 1) + 1

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        u1 = self.up1(s3)
        u2 = self.up2(torch.cat((u1, s2), 1))
        y = self.out(torch.cat((u2, s1), 1))
        return y
