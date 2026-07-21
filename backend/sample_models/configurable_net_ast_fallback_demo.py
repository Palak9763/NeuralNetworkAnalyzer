import torch.nn as nn


class ConfigurableNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
