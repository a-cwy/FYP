import json
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ffnet
import ffevalutil
import npevalutil



#Main
@hydra.main(config_path = "./ff", config_name = "config", version_base = None)
def mainfunc(opt: DictConfig) -> None:
    eval_set = "CIFAR10" #MNIST / CIFAR10
    model = "np"
    
    results = {
        "run1": {
            "pred": [],
            "true": []
        },
        "run2": {
            "pred": [],
            "true": []
        },
        "run3": {
            "pred": [],
            "true": []
        }
    }
    
    test_data = None
    if eval_set == "MNIST":
        test_data = datasets.MNIST("./data/MNIST", train = False, download = True, transform = transforms.ToTensor())
    elif eval_set == "CIFAR10":
        test_data = datasets.CIFAR10("./data/CIFAR10", train = False, download = True, transform = transforms.ToTensor())

    test_loader = DataLoader(test_data, batch_size = 1, shuffle = True)



    if model == "bp":
        path = f"./bp/models/{eval_set}/final"

        #Test loop
        for i in range(1, 4):
            loaded_model = None
            if eval_set == "MNIST":
                loaded_model = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(1600, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            elif eval_set == "CIFAR10":
                loaded_model = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(2304, 700),
                    nn.ReLU(),
                    nn.Linear(700, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            loaded_model.load_state_dict(state_dict = torch.load(f"{path}{i}/bp-best", weights_only = True))
            loaded_model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    out = loaded_model(x)
                    y_hat = torch.argmax(out, dim = -1, keepdim = True)

                    results[f"run{i}"]["pred"].append(y_hat.item())
                    results[f"run{i}"]["true"].append(y.item())

        pass
    elif model == "ff":
        path = f"./ff/models/{eval_set}/final"
        
        #Test loop
        for i in range(1, 4):
            loaded_model = None
            if eval_set == "MNIST":
                loaded_model = ffnet.FFnet(opt, width = 784, hidden_dim = 800)
            elif eval_set == "CIFAR10":
                loaded_model = ffnet.FFnet(opt, width = 3072, hidden_dim = 500)
            
            loaded_model.load_state_dict(state_dict = torch.load(f"{path}{i}/ff-best", weights_only = True))
            loaded_model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    input, label = ffevalutil.getitem(x, y)
                    out = loaded_model(input, label)
                    y_hat = torch.argmax(out, dim = -1, keepdim = True)

                    results[f"run{i}"]["pred"].append(y_hat.item())
                    results[f"run{i}"]["true"].append(y.item())
                    

        pass
    elif model == "np":
        path = f"./np/models/{eval_set}/final"
        
        #Test loop
        for i in range(1, 4):
            loaded_model = None
            if eval_set == "MNIST":
                loaded_cnn = npevalutil.CNNmnist()
            elif eval_set == "CIFAR10":
                loaded_cnn = npevalutil.CNNcifar()

            loaded_cnn.load_state_dict(state_dict = torch.load(f"{path}{i}/np-best/np-checkpoint-cnn", weights_only = True))
            loaded_cnn.eval()
            loaded_blocks = nn.ModuleList([npevalutil.DenoiserBlock(label_dim = 10) for _ in range(10)])
            for index, block in enumerate(loaded_blocks):
                block.load_state_dict(state_dict = torch.load(f"{path}{i}/np-best/np-checkpoint-block-{index}", weights_only = True))
                block.eval()
            
            with torch.no_grad():
                alpha = torch.linspace(1.0, 0.1, 10)

                for x, y in test_loader:
                    z_t = torch.randn(1, 10)
                    
                    x_features = loaded_cnn(x) 
                    for t in reversed(range(10)):
                        u_hat = loaded_blocks[t](x_features, z_t)
                        z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)
                    
                    softmax = nn.functional.softmax(z_t)
                    y_hat = torch.argmax(softmax, dim = -1, keepdim = True).flatten()

                    results[f"run{i}"]["pred"].append(y_hat.item())
                    results[f"run{i}"]["true"].append(y.item())
                    


        pass


    os.makedirs(f"./evals/{model}/{eval_set}", exist_ok = True)
    with open(f"./evals/{model}/{eval_set}/res.json", "w", encoding = 'utf-8') as f:
        f.write(json.dumps(results, ensure_ascii = False, indent = 4))


if __name__ == "__main__":
    mainfunc()