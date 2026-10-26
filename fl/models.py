"""PyTorch models for Federated Learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """Simple CNN for MNIST (2 conv + 2 fc)."""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Net(nn.Module):
    """Compact CNN for CIFAR-10 suited for Wireless Federated Learning.

    Uses GroupNorm instead of BatchNorm to ensure stable federated aggregation
    without running statistic mismatch across non-IID client distributions.
    Dimensioned so model size (~57k params, ~1.83 Mbits) aligns with realistic
    wireless channel capacity (10-50 Mbps) and latency budgets (50-200 ms).
    """
    def __init__(self, num_classes: int = 10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def get_model(dataset_name: str, device: torch.device) -> nn.Module:
    """Factory function."""
    if dataset_name.lower() == "mnist":
        model = MNISTNet()
    elif dataset_name.lower() == "cifar10":
        model = CIFAR10Net()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return model.to(device)
