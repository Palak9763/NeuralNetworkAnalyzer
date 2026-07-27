import torch.nn as nn


class BasicBlock(nn.Module):
    """A standard ResNet BasicBlock: conv-bn-relu-conv-bn, + identity shortcut, then relu."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # residual add - the thing Phase 3 needs to detect
        out = self.relu2(out)
        return out


class MiniResNet(nn.Module):
    """Two stages of repeated identical BasicBlocks, like real ResNet stages."""
    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 16, 7, padding=3)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()

        self.block1 = BasicBlock(16)
        self.block2 = BasicBlock(16)
        self.block3 = BasicBlock(16)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x