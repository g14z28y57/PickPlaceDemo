import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接 + 最终激活
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.stage1 = ResidualBlock(32, 64, 2)
        self.stage2 = ResidualBlock(64, 128, 2)
        self.stage3 = ResidualBlock(128, 256, 2)
        self.stage4 = ResidualBlock(256, 512, 2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始层
        out = self.conv1(x)

        # 四个阶段
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 最终层
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out


if __name__ == '__main__':
    model = ResNet()
    input_tensor = torch.randn(1, 3, 512, 512)
    output = model(input_tensor)