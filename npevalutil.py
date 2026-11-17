from torch import nn
import torch


class DenoiserBlock(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.denoiseblock = nn.Sequential(
            nn.Linear(256 + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, label_dim)
        )

    def forward(self, x, z_t):
        input = torch.cat([x, z_t], dim = 1)
        return self.denoiseblock(input)

class CNNmnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 256)
        )

    def forward(self, x):
        return self.net(x)

class CNNcifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 256)
        )

    def forward(self, x):
        return self.net(x)
