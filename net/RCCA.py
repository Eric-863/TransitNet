import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple
import numpy as np


class ConvMlp(nn.Module):
    """ 使用1x1卷积的MLP模块，支持分组卷积和激活函数配置 """

    def __init__(
            self, in_features, hidden_features=None, out_features=None,
            act_layer=nn.GELU, norm_layer=nn.BatchNorm2d,
            bias=True, drop=0., group=1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1,
                             bias=bias[0], groups=group)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1,
                             bias=bias[1], groups=group)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCCA(nn.Module):
    """ 矩形上下文校准注意力（Rectangular Context Calibration Attention） """

    def __init__(self, inp, kernel_size=1, ratio=1,
                 band_kernel_size=11, square_kernel_size=3,
                 multi_scale=False, scales=(1, 2),
                 use_dwconv=True, norm_layer=nn.BatchNorm2d
                 ):
        super(RCCA, self).__init__()
        self.multi_scale = multi_scale
        self.scales = scales
        self.inp = inp

        # 局部特征提取
        if use_dwconv:
            self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size,
                                       padding=square_kernel_size // 2, groups=inp)
        else:
            self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size,
                                       padding=square_kernel_size // 2)

        # 全局上下文提取
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 多尺度带形卷积配置（修复尺寸问题）
        gc = inp // ratio
        self.excite = nn.ModuleList()
        for scale in scales:
            scaled_band = band_kernel_size * scale
            # 自适应padding确保输出尺寸一致
            padding_h = 0
            padding_w = scaled_band // 2
            self.excite.append(nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, scaled_band),
                          padding=(padding_h, padding_w), groups=gc),
                norm_layer(gc),
                nn.ReLU(inplace=True),
                # 水平卷积后调整尺寸
                nn.Conv2d(gc, inp, kernel_size=(scaled_band, 1),
                          padding=(padding_w, padding_h), groups=gc),
                nn.Sigmoid()
            ))

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # [N, C, H, W]

        # 多尺度上下文融合（添加尺寸对齐）
        if self.multi_scale:
            att = 0
            h, w = x_gather.shape[2], x_gather.shape[3]
            for excite in self.excite:
                # 前向传播
                att_scale = excite(x_gather)
                # 自适应调整尺寸以匹配输入
                att_scale = F.interpolate(att_scale, size=(h, w), mode='bilinear', align_corners=False)
                att += att_scale
            att = att / len(self.excite)
        else:
            att = self.excite[0](x_gather)

        return att

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc
        return out


class RCCAModule(nn.Module):
    """ 基于RCCA的增强型模块（RCCA Module） """

    def __init__(
            self, dim, token_mixer=RCCA,
            norm_layer=nn.BatchNorm2d, mlp_layer=ConvMlp,
            mlp_ratio=2, act_layer=nn.GELU,
            ls_init_value=1e-6, drop_path=0.,
            dw_size=11, square_kernel_size=3,
            ratio=1, use_dynamic_gamma=False,
            use_multi_scale=True, scales=(1, 2),
            visualize=False
    ):
        super().__init__()
        self.visualize = visualize
        self.token_mixer = token_mixer(
            dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
            ratio=ratio, multi_scale=use_multi_scale, scales=scales
        )
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(
            dim, int(mlp_ratio * dim), act_layer=act_layer,
            group=min(dim // 2, 16)  # 分组卷积提升效率
        )

        # 动态缩放因子
        if use_dynamic_gamma:
            self.gamma = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.gamma = nn.Parameter(
                ls_init_value * torch.ones(dim), requires_grad=True
            ) if ls_init_value else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_dynamic_gamma = use_dynamic_gamma

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)

        # 保存中间特征用于可视化
        if self.visualize and self.training:
            self.intermediate = x.clone()

        x = self.mlp(x)

        # 动态尺度调整
        if self.gamma is not None:
            if self.use_dynamic_gamma:
                gamma = self.gamma(x)
                x = x * gamma
            else:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + shortcut

        # 可视化特征图
        if self.visualize and self.training:
            self.visualize_features(x)

        return x

    def visualize_features(self, feature_map):
        """ 可视化中间特征图，用于调试和理解模块行为 """
        if feature_map.dim() == 4:
            b, c, h, w = feature_map.shape
            if b > 1:
                feature_map = feature_map[0]  # 仅可视化第一个样本

            # 选择前16个通道可视化
            n = min(c, 16)
            # 为避免matplotlib显示问题，转换为numpy数组
            feature_map_np = feature_map.detach().cpu().numpy()
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i in range(n):
                ax = axes[i // 4, i % 4]
                ax.imshow(feature_map_np[i], cmap='viridis')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'rcca_features_{np.random.randint(0, 1000)}.png')
            plt.close()


if __name__ == '__main__':
    # 测试不同输入尺寸和配置
    test_cases = [
        (1, 64, 32, 32),  # 常规尺寸
        (1, 32, 64, 64),  # 高分辨率
        (1, 128, 16, 16)  # 低分辨率
    ]

    for shape in test_cases:
        print(f"\n测试输入形状: {shape}")
        input_tensor = torch.randn(shape)

        # 基础配置
        block = RCCAModule(dim=shape[1], drop_path=0.1)
        output = block(input_tensor)
        print(f"基础配置输出形状: {output.shape}")

        # 多尺度配置（修复尺寸问题）
        block_ms = RCCAModule(dim=shape[1], use_multi_scale=True, scales=(1, 2))
        output_ms = block_ms(input_tensor)
        print(f"多尺度配置输出形状: {output_ms.shape}")

        # 动态gamma配置
        block_dg = RCCAModule(dim=shape[1], use_dynamic_gamma=True)
        print(block_ms)
        output_dg = block_dg(input_tensor)
        print(f"动态gamma配置输出形状: {output_dg.shape}")