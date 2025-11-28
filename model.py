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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=3, bias=False)
        self.stage1 = ResidualBlock(32, 64)
        self.stage2 = ResidualBlock(64, 128)
        self.stage3 = ResidualBlock(128, 256)
        self.stage4 = ResidualBlock(256, 512)
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
        out = self.pooling(out)
        out = torch.flatten(out, 1)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.ReLU(inplace=True),
            nn.Linear(dim2, dim1),
            nn.BatchNorm1d(dim1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layers(x) + x
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_layers):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, h_dim)
        self.layers = nn.ModuleList([BottleneckBlock(h_dim, 2 * h_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return x


def main():
    device = "cuda"
    model = ResNet()
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        print(output.shape)


if __name__ == '__main__':
    device = "cuda"
    model = MLP(512, 32, 5, 4)
    model = model.to(device)
    input_tensor = torch.randn(13, 512).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        print(output.shape)
