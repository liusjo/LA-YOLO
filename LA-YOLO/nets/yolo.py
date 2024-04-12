import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny
from nets.attensions import CBAM,CoordAtt,SEAttention
from nets.heads import TSCODE_Detect,TSCODE_Detect_sim
from nets.necks import neck,neck_fine,xfpn,ACMneck,DenseNeck

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', att='cbam', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 3)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self. backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)

        #Attantions
        if att=='CA':
            self.att3=CoordAtt( base_channels * 16,base_channels * 16)
            self.att2=CoordAtt( base_channels * 8,base_channels * 8)
            self.att1=CoordAtt( base_channels * 4,base_channels * 4)
        elif att=='SE':
            self.att3=SEAttention(channel =base_channels * 16)
            self.att2=SEAttention(channel =base_channels * 8)
            self.att1=SEAttention(channel =base_channels * 4)
        else:
            self.att3 = CBAM(base_channels * 16)
            self.att2 = CBAM(base_channels * 8)
            self.att1 = CBAM(base_channels * 4)


        # 新neck
        # self.neck = neck_fine(base_channels)
        self.xfpn = xfpn(base_channels)
        # self.acmneck = ACMneck(base_channels)
        # self.densenet = DenseNeck(base_channels)


        # NECK
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv_new1              = Conv(base_channels * 16, base_channels * 8, 1, 1)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)
        self.conv_new2              = Conv(base_channels * 32, base_channels * 16, 1, 1)
        self.conv_new21             = Conv(base_channels * 36, base_channels * 16, 1, 1)

        # 新增的p2部分操作
        self.conv_for_feat1         = Conv(base_channels * 4, base_channels * 2, 1, 1)
        self.conv3_for_upsample3 = C3(base_channels * 4, base_channels * 2, base_depth, shortcut=False)
        self.down_sample3 = Conv(base_channels * 2, base_channels * 2, 3, 2)
        self.conv_channel1 = Conv(base_channels * 8, base_channels * 4, 1, 1)

        self.conv_channel2 = Conv(base_channels * 14, base_channels * 8, 1, 1)
        self.conv_channel3 = Conv(base_channels * 26, base_channels * 16, 1, 1)

        self.conv3_for_downsample0 = C3(base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        # self.conv3_for_downsample0 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        #新增DHEAD需要做的的操作
        self.down_sample0 = Conv(base_channels * 16, base_channels * 16, 3, 2)


        #HEAD
        # self.yolo_head_P2 = nn.Conv2d(base_channels * 2, len(anchors_mask[3]) * (5 + num_classes), 1)
        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

        # DHEAD1
        self.head_P3=TSCODE_Detect((base_channels * 2,base_channels * 4,base_channels * 8)
                                  ,num_classes
                                  ,len(anchors_mask[2]))
        self.head_P4=TSCODE_Detect((base_channels * 4,base_channels * 8,base_channels * 16)
                                  ,num_classes
                                  ,len(anchors_mask[1]))
        self.head_P5=TSCODE_Detect((base_channels * 8,base_channels * 16,base_channels * 16)
                                  ,num_classes
                                  ,len(anchors_mask[0]))
        # # DHEAD2
        # self.head_P3 = TSCODE_Detect_sim((base_channels * 2, base_channels * 4, base_channels * 8)
        #                              , num_classes
        #                              , len(anchors_mask[2]))
        # self.head_P4 = TSCODE_Detect_sim((base_channels * 4, base_channels * 8, base_channels * 16)
        #                              , num_classes
        #                              , len(anchors_mask[1]))
        # self.head_P5 = TSCODE_Detect_sim((base_channels * 8, base_channels * 16, base_channels * 16)
        #                              , num_classes
        #                              , len(anchors_mask[0]))



    def forward(self, x):
        #  backbone
        feat0, feat1, feat2, feat3 = self.backbone(x)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 初始neck ——begin——
        # Att
        # feat3 = self.att3(feat3)
        # feat2 = self.att2(feat2)
        # feat1 = self.att1(feat1)
        # P6 = self.down_sample0(feat3)
        # # 20, 20, 1024 -> 20, 20, 512
        # P5 = self.conv_for_feat3(feat3)
        # # 20, 20, 512 -> 40, 40, 512
        # P5_upsample = self.upsample(P5)
        # # 40, 40, 512 -> 40, 40, 1024
        # P4 = torch.cat([P5_upsample, feat2], 1)
        # # 40, 40, 1024 -> 40, 40, 512
        # P4 = self.conv3_for_upsample1(P4)  # csplayer
        #
        # # 40, 40, 512 -> 40, 40, 256
        # P4 = self.conv_for_feat2(P4)
        # # 40, 40, 256 -> 80, 80, 256
        # P4_upsample = self.upsample(P4)
        # # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        # P3 = torch.cat([P4_upsample, feat1], 1)
        # # 80, 80, 512 -> 80, 80, 256
        # P3 = self.conv3_for_upsample2(P3)  # csplayer
        #
        # P2 = feat0
        #
        # # 80, 80, 256 -> 40, 40, 256
        # P3_downsample = self.down_sample1(P3)
        # # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        # P4 = torch.cat([P3_downsample, P4], 1)
        # # 40, 40, 512 -> 40, 40, 512
        # P4 = self.conv3_for_downsample1(P4)  # csplayer
        #
        # # 40, 40, 512 -> 20, 20, 512
        # P4_downsample = self.down_sample2(P4)
        # # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        # P5 = torch.cat([P4_downsample, P5], 1)
        # # 20, 20, 1024 -> 20, 20, 1024
        # P5 = self.conv3_for_downsample2(P5)  # csplayer
        #
        # out2 = self.head_P3((P2, P3, P4))
        # out1 = self.head_P4((P3, P4, P5))
        # out0 = self.head_P5((P4, P5, P6))
        #
        # # # ---------------------------------------------------#
        # # #   第三个特征层
        # # #   y3=(batch_size,75,80,80)
        # # # ---------------------------------------------------#
        # # out2 = self.yolo_head_P3(P3)
        # # # ---------------------------------------------------#
        # # #   第二个特征层
        # # #   y2=(batch_size,75,40,40)
        # # # ---------------------------------------------------#
        # # out1 = self.yolo_head_P4(P4)
        # # # ---------------------------------------------------#
        # # #   第一个特征层
        # # #   y1=(batch_size,75,20,20)
        # # # ---------------------------------------------------#
        # # out0 = self.yolo_head_P5(P5)
        #
        # return out0, out1, out2
        # 初始neck ——end——


        # # neck fine ——begin——
        # p0, p1, p2, p3 = self.neck(feat0, feat1, feat2, feat3)
        # p4 = self.down_sample0(p3)
        # out2 = self.head_P3((p0, p1, p2))
        # out1 = self.head_P4((p1, p2, p3))
        # out0 = self.head_P5((p2, p3, p4))
        # return out0, out1, out2
        # # neck fine ——end——


        # x-fpn ——begin——
        # # Att
        # feat1 = self.att1(feat1)
        # feat2 = self.att2(feat2)
        # feat3 = self.att3(feat3)

        x5 = feat3
        P6 = self.down_sample0(feat3)
        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)  # csplayer

        x4 = P4

        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)  # csplayer

        x3 = P3

        # 80, 80, 256 -> 80, 80, 128
        P3 = self.conv_for_feat1(P3)
        # 80, 80, 128 -> 160, 160, 128
        P3_upsample = self.upsample(P3)
        # 160, 160, 128 cat 160, 160, 128 -> 160, 160, 256
        P2 = torch.cat([P3_upsample, feat0], 1)
        # 160, 160, 256 -> 160, 160, 128
        P2 = self.conv3_for_upsample3(P2)  # csplayer

        x2 = P2

        xx2,xx3,xx4,xx5 = self.xfpn(x2,x3,x4,x5)

        # 160, 160, 128 -> 80, 80, 128
        xx2_downsample = self.down_sample3(xx2)
        # 80, 80, 256 -> 80, 80, 128
        xx3 = self.conv_for_feat1(xx3)
        # 80, 80, 128 cat 80, 80, 128 -> 80, 80, 256
        xx3 = torch.cat([xx2_downsample, xx3], 1)
        # 80, 80, 256 -> 80, 80, 256
        xx3 = self.conv3_for_downsample0(xx3)  # csplayer


        # 80, 80, 256 -> 40, 40, 256
        xx3_downsample = self.down_sample1(xx3)
        # 40, 40, 512 -> 40, 40, 256
        xx4 = self.conv_for_feat2(xx4)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        xx4 = torch.cat([xx3_downsample, xx4], 1)
        # 40, 40, 512 -> 40, 40, 512
        xx4 = self.conv3_for_downsample1(xx4)  # csplayer

        # 40, 40, 512 -> 20, 20, 512
        xx4_downsample = self.down_sample2(xx4)
        # 20, 20, 1024 -> 20, 20, 512
        xx5 = self.conv_for_feat3(xx5)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        xx5 = torch.cat([xx4_downsample, xx5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        xx5 = self.conv3_for_downsample2(xx5)  # csplayer

        # out2 = self.head_P3((xx2, xx3, xx4))
        # out1 = self.head_P4((xx3, xx4, xx5))
        # out0 = self.head_P5((xx4, xx5, P6))

        # out3 = self.yolo_head_P2(xx2)
        out2 = self.yolo_head_P3(xx3)
        out1 = self.yolo_head_P4(xx4)
        out0 = self.yolo_head_P5(xx5)
        return out0, out1, out2
        # x-fpn ——end——

        # acm net ——begin——
        # p0, p1, p2, p3 = self.acmneck(feat0, feat1, feat2, feat3)
        # p4 = self.down_sample0(p3)
        # # out2 = self.head_P3((p0, p1, p2))
        # # out1 = self.head_P4((p1, p2, p3))
        # # out0 = self.head_P5((p2, p3, p4))
        # out2 = self.yolo_head_P3(p1)
        # out1 = self.yolo_head_P4(p2)
        # out0 = self.yolo_head_P5(p3)
        # return out0, out1, out2
        # acm net ——end——

        # dense net ——begin——
        # p0, p1, p2, p3 = self.densenet(feat0, feat1, feat2, feat3)
        # p4 = self.down_sample0(p3)
        # out2 = self.head_P3((p0, p1, p2))
        # out1 = self.head_P4((p1, p2, p3))
        # out0 = self.head_P5((p2, p3, p4))
        # # out2 = self.yolo_head_P3(p1)
        # # out1 = self.yolo_head_P4(p2)
        # # out0 = self.yolo_head_P5(p3)
        # return out0, out1, out2
        # dense net ——end——


if __name__ == '__main__':
    # 输入的通道数要和embed_dim一致
    x = torch.randn((4, 640, 24, 24))
    model = YoloBody()
    # 测试模型的大小
    # device = torch.device('cuda:0')
    # input = x.to(device)
    model.eval()
    # model = model.to(device)
    # x = model(input)
    x = model(x)
    print(x.shape)