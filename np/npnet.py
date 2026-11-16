from torch import nn
import torch


#Denoiser block that progressively denoises label towards prediction label
class DenoiserBlock(nn.Module):
    def __init__(self, label_dim, hidden_size = 256):
        super().__init__()
        self.denoiseblock = nn.Sequential(
            nn.Linear(256 + label_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, label_dim)
        )

    def forward(self, x, z_t):
        input = torch.cat([x, z_t], dim = 1)
        return self.denoiseblock(input)



#CNN to extract features from image
class CNN(nn.Module):
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
            #MNIST: 1600
            #CIFAR10: 2304
        )

    def forward(self, x):
        return self.net(x)
