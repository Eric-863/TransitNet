import torch
import torch.nn as nn

class BiFusionAttention(nn.Module):
    """
    BiFusion Attention (BFA) 模块 - 融合空间和通道维度的注意力机制
    通过多尺度空间分支和轻量化通道注意力实现高效语义分割
    """

    def __init__(self, channel, base_kernel=5, scales=3,
                 fusion_factor=1.0, reduction_ratio=32):
        super().__init__()
        self.scales = scales
        self.reduction_ratio = reduction_ratio

        # 共享的特征提取层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False)
        )

        # 多尺度空间注意力分支
        self.spatial_branches = nn.ModuleList()
        for i in range(scales):
            kernel_size = base_kernel + i * 2  # 递增的卷积核大小
            self.spatial_branches.append(nn.Sequential(
                nn.Conv2d(channel // 2, channel // 2, kernel_size=kernel_size,
                          padding=kernel_size // 2, groups=channel // 2, bias=False),
                nn.BatchNorm2d(channel // 2),
                nn.Sigmoid()
            ))

        # 轻量化通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // 2, max(8, (channel // 2) // reduction_ratio), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, (channel // 2) // reduction_ratio), channel // 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channel // 2 * scales, channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        # 融合因子
        self.fusion_factor = nn.Parameter(torch.tensor(fusion_factor))

    def forward(self, x):
        # 特征提取
        shared_features = self.shared_conv(x)

        # 多尺度空间注意力
        spatial_attention_maps = []
        for branch in self.spatial_branches:
            spatial_map = branch(shared_features)
            spatial_attention_maps.append(spatial_map)

        # 通道注意力
        channel_map = self.channel_attn(shared_features)

        # 融合多尺度特征
        fused_features = torch.cat([feat * channel_map for feat in spatial_attention_maps], dim=1)
        fused_features = self.fusion(fused_features)

        # 生成最终注意力图
        attention_map = self.output(fused_features)

        # 应用注意力并结合残差连接
        return x + self.fusion_factor * x * attention_map
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width]输入张量
    input = torch.randn(1, 32, 256, 256)
    print(f"输入形状: {input.shape}")
    # 初始化模块
    ela = BiFusionAttention(channel=32, base_kernel=7)
    # 前向传播
    output = ela(input)
    # 打印出输出张量的形状，它将与输入形状相匹配
    print(f"输出形状: {output.shape}")

