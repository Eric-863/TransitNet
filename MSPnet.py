import torch
import torch.nn.functional as F
from torch import nn
from nets.StarNet import starnet_s4, starnet_s3, starnet_s2, starnet_s1
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.RCCA import RCCAModule
from nets.BFA import BiFusionAttention
# from nets.MSAA import MSAA
# from nets.ELA import EfficientLocalizationAttention
# from nets.PagFM import PagFM

# import time
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def draw_features(width, height, x, savename):
#     tic = time.time()
#     fig = plt.figure(figsize=(16, 16))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
#     for i in range(width*height):
#         plt.subplot(height, width, i + 1)
#         plt.axis("off")
#         img = x[0, i, :, :]
#         pmin = np.min(img)
#         pmax = np.max(img)
#         img = (img - pmin)/(pmax - pmin + 0.000001)
#         plt.imshow(img, cmap='viridis')
#         print("{}/{}".format(i, width*height))
#     # 确保保存路径存在
#     save_dir = os.path.dirname(savename)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     fig.savefig(savename, dpi=300)
#     fig.clf()
#     plt.close()
#     print("time:{}".format(time.time()-tic))


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        # --------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x


# 深度可分离卷积
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            DepthWiseConv(in_channels + (out_channels * len(pool_sizes)), out_channels),  # 替换卷积
            # nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1,
            #           bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = DepthWiseConv(in_channels, out_channels)  # 替换卷积
        # conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True, name=""):
        super(PSPNet, self).__init__()
        # self.savepath = "D:/CV/pspnet-pytorch-master/heatmaps".format(name)
        # self.image_counter = 1  # 初始化计数器
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320

        elif backbone == "starnet1":
            self.backbone = starnet_s1(pretrained)
            aux_channel = 96
            out_channel = 192

        elif backbone == "starnet2":
            self.backbone = starnet_s2(pretrained)
            aux_channel = 128
            out_channel = 256

        elif backbone == "starnet3":
            self.backbone = starnet_s3(pretrained)
            aux_channel = 128
            out_channel = 256

        elif backbone == "starnet4":
            self.backbone = starnet_s4(pretrained)
            aux_channel = 128
            out_channel = 256
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        # self.ela = EfficientLocalizationAttention(aux_channel, 5)
        # self.ela1 = EfficientLocalizationAttention(out_channel, 7)
        self.rcca = RCCAModule(dim=out_channel)
        self.bfa = BiFusionAttention(out_channel, 5)
        # self.pld = PagFM(aux_channel, 16)
        # self.pld1 = PagFM(out_channel, 16)

        #         self.msaa = MSAA(out_channel, out_channel)

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                DepthWiseConv(aux_channel, out_channel // 8),  # 替换卷积
                # nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        # 生成唯一的文件名
        # savename = os.path.join(self.savepath, f"f{self.image_counter}.png")
        # draw_features(16, 16, x.cpu().detach().numpy(), savename)
        # print(savename)
        # self.image_counter += 1  # 计数器加1
        # x1 = self.ela(x_aux)
        # x2 = self.ela1(x)
        x = self.rcca(x)
        x = self.bfa(x)
        # x_aux = self.pld(x1, x_aux)
        # x = self.pld1(x2, x)
        # print(x.shape)

        #         x = self.msaa(x, x, x)

        output = self.master_branch(x)

        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()


if __name__ == "__main__":
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 starnet模型
    backbone = "starnet1"
    downsample_factor = 16
    pretrained = True
    model = PSPNet(num_classes=21, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained).to(
        device)
    print("Model loaded successfully.")
    # 模拟输入数据（batch_size=1, 3通道图像，大小为224x224）
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    # 模型推理
    with torch.no_grad():  # 关闭梯度计算
        model.eval()  # 设置模型为评估模式
        output = model(input_tensor)
    # 打印输出结果形状
    print(f"Output shape: {output.shape}")