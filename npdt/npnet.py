from torch import nn
import torch



class CNNmnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        return self.model(x)



class CNNcifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 64),
            nn.BatchNorm1d(64)
        )

    def forward(self, x):
        return self.model(x)
    


class LabelEmbedder(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.preskip = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.postskip = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64)
        )

    def forward(self, x):
        preskip_out = self.preskip(x)
        postskip_out = self.postskip(preskip_out)
        out = torch.add(preskip_out, postskip_out)
        return out




class DenoiserBlock(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, label_dim)
        )

    def forward(self, x):
        return self.model(x)