from einops import rearrange
import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny
from nets.attensions import CBAM, CoordAtt, SEAttention


class TSCODE_Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, ch, nc=80, na=3, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.na = na  # number of anchors
        self.ph, self.pw = 2, 2

        self.SCE = SCE(ch)
        self.DPE = DPE(ch, ch[1])
        self.m_cls = nn.Sequential(Conv(ch[2] * 2, ch[2], 1), Conv(ch[2], ch[2], 3),
                                   nn.Conv2d(ch[2], self.na * self.nc * 4, 1))
        self.m_reg_conf = nn.Sequential(Conv(ch[1], ch[1], 3), Conv(ch[1], ch[1], 3), nn.Conv2d(ch[1], self.na * 5, 1))
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x_):
        bs, _, ny, nx = x_[1].shape

        x_sce, x_dpe = self.SCE(x_[1:]), self.DPE(x_)
        x_cls = rearrange(self.m_cls(x_sce), 'bs (na ph pw nc) h w -> bs (na nc) (h ph) (w pw)', na=self.na, ph=self.ph,
                          pw=self.pw, nc=self.nc)
        # x_cls = x_cls.permute(0, 1, 3, 4, 2).contiguous()

        x_reg_conf = self.m_reg_conf(x_dpe)
        x = torch.cat([x_reg_conf, x_cls], dim=1)
        return x


class TSCODE_Detect_sim(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, ch, nc=80, na=3, inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.na = na  # number of anchors
        self.ph, self.pw = 2, 2

        self.down = Conv(ch[0], ch[1], k=3, s=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cls = Conv(ch[1] * 2, self.na * self.nc, 1)
        self.reg = Conv(ch[1] + ch[2], self.na * 5, 1, act=nn.Sigmoid())

    def forward(self, x_):
        x0 = self.down(x_[0])
        cls = self.cls(torch.cat([x0, x_[1]], dim=1))
        x2 = self.up(x_[2])
        reg = self.reg(torch.cat([x2, x_[1]], dim=1))
        x = torch.cat([reg, cls], dim=1)
        return x


class SCE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = Conv(ch[1], ch[2], k=3, s=2)

    def forward(self, x):
        x_p1, x_p2 = x
        x = torch.cat([self.down(x_p1), x_p2], 1)
        return x


class DPE(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.adjust_channel_forp1 = Conv(c1[0], c2, k=1)
        self.adjust_channel_forp2 = Conv(c1[1], c2, k=1)

        self.up_forp2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(c2, c2, k=1)
        )
        self.up_forp3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(c1[2], c2, k=1)
        )
        self.down = Conv(c2, c2, k=3, s=2)
        self.middle = Conv(c2, c2, k=1)

    def forward(self, x):
        x_p2 = self.adjust_channel_forp2(x[1])
        x_p1 = self.adjust_channel_forp1(x[0]) + self.up_forp2(x_p2)
        x_p1 = self.down(x_p1)

        x_p3 = self.up_forp3(x[2])

        return x_p1 + x_p2 + x_p3