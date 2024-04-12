import torch
import torch.nn as nn
from .attensions import CBAM


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))


class Og1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Og1, self).__init__()
        self.cv1 = Conv(int(c1 / 2), c1, 3, 2, p=1)
        self.link1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(c1, int(c1 / 4)),
            Conv(int(c1 / 4), c1, act=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(c1, eps=0.001, momentum=0.03),
            SiLU()
        )
        self.cv2 = Conv(c1, c1)

    def forward(self, x0, x1):
        x0 = self.cv1(x0)
        x1 = self.link1(x1)
        x = x0 * x1
        x = self.cv2(x)
        return x


class Og2(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Og2, self).__init__()
        self.cv1 = Conv(c1 * 2, c1, 3, p=1)
        self.link1 = nn.Sequential(
            Conv(c1, int(c1 / 4)),
            Conv(int(c1 / 4), c1, act=False),
            nn.Sigmoid(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c1 * 2, c1)

    def forward(self, x0, x1):
        x0 = self.link1(x0)
        x1 = self.cv1(x1)
        x1 = self.up(x1)
        x = torch.cat([x0, x1], 1)
        x = self.cv2(x)
        return x


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1
            )
        )


class Focuss(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, T=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focuss, self).__init__()
        self.conv = Conv(c1 * (4 ** T), c2, k, s, p, g, act)
        self.T = T

    def forward(self, x):
        for i in range(self.T):
            x = torch.cat([
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], 1)
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class neck(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.f1 = Focuss(base_channels, base_channels * 2, k=3, T=1)
        self.f2 = Focuss(base_channels, base_channels * 4, k=3, T=2)
        self.f3 = Focuss(base_channels, base_channels * 8, k=3, T=3)

        self.ccat1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, 1)
        self.ccat2 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, 1)
        self.ccat3 = nn.Conv2d(base_channels * 16, base_channels * 8, 1, 1)

        self.c1 = nn.Conv2d(base_channels * 8, base_channels * 16, 3, 2, autopad(3))

    def forward(self, f0, f1, f2, f3):
        p0 = self.f1(f0)
        p1 = self.f2(f0)
        p2 = self.f3(f0)

        p0 = torch.cat([p0, f1], 1)
        p0 = self.ccat1(p0)

        p1 = torch.cat([p1, f2], 1)
        p1 = self.ccat2(p1)

        p2 = torch.cat([p2, f3], 1)
        p2 = self.ccat3(p2)

        p3 = self.c1(p2)

        return p0, p1, p2, p3


class neck_fine(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.att3 = CBAM(base_channels * 16)
        self.att2 = CBAM(base_channels * 8)
        self.att1 = CBAM(base_channels * 4)

        self.og2_1 = Og2(base_channels * 2)
        self.og2_2 = Og2(base_channels * 4)
        self.og2_3 = Og2(base_channels * 8)

        self.og1_1 = Og1(base_channels * 4)
        self.og1_2 = Og1(base_channels * 8)
        self.og1_3 = Og1(base_channels * 16)

        self.cat1 = Conv(base_channels * 8, base_channels * 4)
        self.cat2 = Conv(base_channels * 16, base_channels * 8)

        self.ccv0 = Conv(base_channels * 4, base_channels * 2)
        self.ccv1 = Conv(base_channels * 8, base_channels * 4)
        self.ccv2 = Conv(base_channels * 16, base_channels * 8)
        self.ccv3 = Conv(base_channels * 32, base_channels * 16)

        self.c3_0 = C3(base_channels * 4, base_channels * 2)
        self.c3_1 = C3(base_channels * 8, base_channels * 4)
        self.c3_2 = C3(base_channels * 16, base_channels * 8)

        # self.cv0=nn.Conv2d(base_channels*2, base_channels*2, 1, 1)
        self.cv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, 1)
        self.cv2 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, 1)
        self.cv3 = nn.Conv2d(base_channels * 16, base_channels * 8, 1, 1)

    def forward(self, f0, f1, f2, f3):
        f1 = self.att1(f1)
        f2 = self.att2(f2)
        f3 = self.att3(f3)

        p0 = self.og2_1(f0, f1)
        p0 = self.ccv0(torch.cat([p0, f0], 1))

        p1_1 = self.og1_1(f0, f1)
        p1_2 = self.og2_2(f1, f2)
        # p1=p1_1+p1_2   相加
        p1 = self.cat1(torch.cat([p1_1, p1_2], 1))  # cat
        p1 = self.ccv1(torch.cat([p1, f1], 1))

        p2_1 = self.og1_2(f1, f2)
        p2_2 = self.og2_3(f2, f3)
        # p2=p2_1+p2_2   #相加
        p2 = self.cat2(torch.cat([p2_1, p2_2], 1))  # cat
        p2 = self.ccv2(torch.cat([p2, f2], 1))

        p3 = self.og1_3(f2, f3)
        p3 = self.ccv3(torch.cat([p3, f3], 1))

        p3_up = self.up(self.cv3(p3))
        p2 = self.c3_2(torch.cat([p2, p3_up], 1))

        p2_up = self.up(self.cv2(p2))
        p1 = self.c3_1(torch.cat([p1, p2_up], 1))

        p1_up = self.up(self.cv1(p1))
        p0 = self.c3_0(torch.cat([p0, p1_up], 1))

        return p0, p1, p2, p3


class ACMneck(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.att3 = CBAM(base_channels * 16)
        self.att2 = CBAM(base_channels * 8)
        self.att1 = CBAM(base_channels * 4)
        self.ACM3 = ACM(base_channels * 8, base_channels * 16)
        self.ACM2 = ACM(base_channels * 4, base_channels * 8)
        self.ACM1 = ACM(base_channels * 2, base_channels * 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, 3, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, 3, shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, 3, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, 3, shortcut=False)

        self.down_sample0 = Conv(base_channels * 16, base_channels * 16, 3, 2)
        self.conv3_for_downsample0 = C3(base_channels * 16, base_channels * 16, 3, shortcut=False)

    def forward(self, f0, f1, f2, f3):
        feat3 = self.att3(f3)
        feat2 = self.att2(f2)
        feat1 = self.att1(f1)

        feat2 = self.ACM3(feat2, feat3)
        feat1 = self.ACM2(feat1, feat2)
        feat0 = self.ACM1(f0, feat1)

        # 20,20,1024->10,10,1024
        P6 = self.down_sample0(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)
        # 160,160,128
        P2 = feat0

        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)
        return P2, P3, P4, P5


class ACM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.adap = Conv(c2, c1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(c1, int(c1 / 4), act=nn.ReLU()),
            Conv(int(c1 / 4), c1)
        )
        self.bottomup = nn.Sequential(
            Conv(c1, int(c1 / 4), act=nn.ReLU()),
            Conv(int(c1 / 4), c1)
        )
        self.post = Conv(c1, c1, k=3)

    def forward(self, x0, x1):
        x1 = self.up(x1)
        x1 = self.adap(x1)
        f0 = self.bottomup(x0)
        f1 = self.topdown(x1)
        f = x0 * f1 + f0 * x1
        f = self.post(f)
        return f


class cfb(nn.Module):
    # 传入高层的c
    def __init__(self, c1, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(cfb, self).__init__()
        self.cv1 = Conv(int(c1 / 2), c1, 3, 2, p=1)
        self.link1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(c1, int(c1 / 4)),
            Conv(int(c1 / 4), c1, act=False),
            nn.Sigmoid(),
        )
        self.cv2 = Conv(c1, c1)

    def forward(self, low, high):
        low = self.cv1(low)
        low = self.link1(low)
        x = low * high
        x = self.cv2(x)
        return x


class rfb(nn.Module):
    # 传入低层的c
    def __init__(self, c1, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(rfb, self).__init__()
        self.cv1 = Conv(c1 * 2, c1, 3, p=1)
        self.link1 = nn.Sequential(
            Conv(2 * c1, int(c1 / 2)),
            Conv(int(c1 / 2), 2 * c1, act=False),
            nn.Sigmoid(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.cv2 = Conv(c1 * 2, c1)

    def forward(self, low, high):
        high = self.link1(high)
        high = self.cv1(high)
        high = self.up(high)
        x = torch.cat([low, high], 1)
        x = self.cv2(x)
        return x


class xfpn(nn.Module):
    def __init__(self, base_channels):
        super().__init__()

        print("xfpn")
        # 第一步:
        self.cfb_1 = cfb(base_channels * 4)
        self.cfb_2 = cfb(base_channels * 8)
        self.cfb_3 = cfb(base_channels * 16)

        self.rfb_1 = rfb(base_channels * 2)
        self.rfb_2 = rfb(base_channels * 4)
        self.rfb_3 = rfb(base_channels * 8)

        # 第二步：
        self.cat1 = Conv(base_channels * 8, base_channels * 4)
        self.cat2 = Conv(base_channels * 16, base_channels * 8)

        # 第三步：残差链接
        self.res0 = Conv(base_channels * 4, base_channels * 2)
        self.res1 = Conv(base_channels * 8, base_channels * 4)
        self.res2 = Conv(base_channels * 16, base_channels * 8)
        self.res3 = Conv(base_channels * 32, base_channels * 16)

    def forward(self, f0, f1, f2, f3):
        # 128/256/512/1024

        # 低层信息不变为基准
        rfb_p2 = self.rfb_1(f0, f1)
        rfb_p3 = self.rfb_2(f1, f2)
        rfb_p4 = self.rfb_3(f2, f3)

        # 高层信息不变为基准
        cfb_p3 = self.cfb_1(f0, f1)
        cfb_p4 = self.cfb_2(f1, f2)
        cfb_p5 = self.cfb_3(f2, f3)

        # 中间两层信息拼接
        p3 = self.cat1(torch.cat([rfb_p3, cfb_p3], 1))
        p4 = self.cat2(torch.cat([rfb_p4, cfb_p4], 1))

        # 残差链接
        p2 = self.res0(torch.cat([rfb_p2, f0], 1))
        p3 = self.res1(torch.cat([p3, f1], 1))
        p4 = self.res2(torch.cat([p4, f2], 1))
        p5 = self.res3(torch.cat([cfb_p5, f3], 1))

        return p2, p3, p4, p5


class DenseNeck(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.att3 = CBAM(base_channels * 16)
        self.att2 = CBAM(base_channels * 8)
        self.att1 = CBAM(base_channels * 4)
        self.att0 = CBAM(base_channels * 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, 3, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, 3, shortcut=False)

        self.conv_for_feat1 = Conv(base_channels * 4, base_channels * 2, 1, 1)
        self.conv3_for_upsample3 = C3(base_channels * 4, base_channels * 2, 3, shortcut=False)

        self.down_sample0 = Conv(base_channels * 2, base_channels * 2, 3, 2)
        self.conv3_for_downsample0 = C3(base_channels * 4, base_channels * 4, 3, shortcut=False)

        self.down_sample1 = Conv(base_channels * 2, base_channels * 4, 3, 4)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, 3, shortcut=False)

        self.down_sample2 = Conv(base_channels * 2, base_channels * 8, 3, 8)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, 3, shortcut=False)

    def forward(self, f0, f1, f2, f3):
        feat3 = self.att3(f3)
        feat2 = self.att2(f2)
        feat1 = self.att1(f1)
        feat0 = self.att0(f0)

        # 20,20,1024->10,10,1024
        # P6          =self.down_sample0 (feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)
        # 160,160,128
        P3 = self.conv_for_feat1(P3)
        # 40, 40, 256 -> 80, 80, 256
        P3_upsample = self.upsample(P3)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P2 = torch.cat([P3_upsample, feat0], 1)
        # 80, 80, 256 -> 80, 80, 128
        P2 = self.conv3_for_upsample3(P2)

        # 80, 80, 128 -> 40, 40, 128
        P3_downsample = self.down_sample0(P2)
        # 40, 40, 128 cat 40, 40, 128 -> 40, 40, 256
        P3 = torch.cat([P3_downsample, P3], 1)
        # 40, 40, 256 -> 40, 40, 256
        P3 = self.conv3_for_downsample0(P3)

        # 80, 80, 128 -> 40, 40, 256
        P4_downsample = self.down_sample1(P2)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P4_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 128 -> 20, 20, 512
        P5_downsample = self.down_sample2(P2)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P5_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        return P2, P3, P4, P5


if __name__ == '__main__':
    # 输入的通道数要和embed_dim一致
    x1 = torch.randn((4, 128, 160, 160))
    x2 = torch.randn((4, 256, 80, 80))
    x3 = torch.randn((4, 512, 40, 40))
    x4 = torch.randn((4, 1024, 20, 20))
    model = xfpn(base_channels=64)
    # 测试模型的大小
    model.eval()
    x1, x2, x3, x4 = model(x1, x2, x3, x4)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)