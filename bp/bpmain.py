from datetime import timedelta
import random
import json
import os
import time
import tracemalloc
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, read_write
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset



#Trainer
def train(opt, train_loader, val_loader, model, optimizer, start_time):
    loss_acc_history = {"loss": [], "accuracy": []}
    val_loss_best = [0.0, 0]
    best_model = model
    scaler = torch.amp.GradScaler()

    epoch = 0
    while(time.time() - start_time <= 600):
    # for epoch in range(opt.training.epochs):
        epoch_loss = 0.0
        batch_count = 0

        #Learning rate decay
        # lr = opt.training.learning_rate
        # if epoch > (opt.training.epochs // 2):
        #     lr = opt.training.learning_rate * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
        # optimizer.param_groups[0]["lr"] = lr

        for x, y in train_loader:
            current_batch_size = x.shape[0]
            
            x = x.to("cuda")

            with torch.amp.autocast(device_type = 'cuda', dtype = torch.float16):
                y_pred = model(x) 
                y_onehot = nn.functional.one_hot(y, num_classes = 10).float()
        
                CELoss = nn.CrossEntropyLoss()
                loss = CELoss(y_pred, y_onehot)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_count += 1

        #Epoch summary
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch}")
        print(f"Average loss {avg_loss}")

        #Validation
        val_loss_acc = validate(opt, val_loader, model)
        print("Validation: ", val_loss_acc)
        loss_acc_history["loss"].append(val_loss_acc["loss"])
        loss_acc_history["accuracy"].append(val_loss_acc["accuracy"])

        if val_loss_acc["accuracy"] > val_loss_best[0]:
            val_loss_best[0] = val_loss_acc["accuracy"]
            val_loss_best[1] = epoch

            # save_dir = "./bp/models/MNIST/final3"
            # torch.save(model.state_dict(), f"{save_dir}/bp-checkpoint-epoch-{val_loss_best[1]}")

        #Early stopper
        # if epoch > val_loss_best[1] + opt.training.early_stop_threshold:
        #     break

        epoch += 1

    return loss_acc_history, val_loss_best, best_model



#Validator
def validate(opt, val_loader, model):
    total_loss = 0.0
    correct_predicitons = 0
    sample_count = 0

    for x, y in val_loader:
        current_batch_size = x.shape[0]

        x = x.to("cuda")
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.float16):
            y_pred = model(x) 

            output = torch.argmax(y_pred, dim = -1, keepdim = True).flatten()
            y_onehot = nn.functional.one_hot(y, num_classes = 10).float()

            CELoss = nn.CrossEntropyLoss()
            loss = CELoss(y_pred, y_onehot).item()
        total_loss += loss

        correct_predicitons += torch.sum(output == y).item()
        sample_count += current_batch_size

    val_loss_acc = {
        "loss": total_loss / sample_count,
        "accuracy": correct_predicitons / sample_count
    }
    
    return val_loss_acc



#Hydra config
@hydra.main(config_path = ".", config_name = "config", version_base = None)
def mainfunc(opt: DictConfig) -> None:
    torch.set_default_device("cuda")
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    #Resource use logging
    start_time = time.time()
    tracemalloc.start()

    #Dataset
    mnist = datasets.MNIST(root = "./data/MNIST/", download = True, train = True, transform = transforms.ToTensor())
    train_data = Subset(mnist, range(50000))
    val_data = Subset(mnist, range(50000, 60000))
    test_data = datasets.MNIST(root = "./data/MNIST/", download = True, train = False, transform = transforms.ToTensor())

    param_list = {
        "learning_rate": [0.0005],
        "weight_decay": [0.001]
    }
    # param_list = {
    #     "learning_rate": [5e-4, 1e-3, 5e-3],
    #     "weight_decay": [5e-4, 1e-3, 5e-3]
    # }

    #Hyperparam tuning loop
    for lr_index in range(len(param_list["learning_rate"])):
        for wd_index in range(len(param_list["weight_decay"])):
            with read_write(opt):
                opt.training.learning_rate = param_list["learning_rate"][lr_index]
                opt.training.weight_decay = param_list["weight_decay"][wd_index]

            model = nn.Sequential(
                nn.Conv2d(1, 4, 3, 1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Flatten(),
                nn.Linear(144, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 10),
                #MNIST: 1600
                #CIFAR10: 2304
            ).to("cuda")
            optimizer = optim.Adam(model.parameters(), lr = opt.training.learning_rate)

            total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total trainable parameters: ", total_param)

            #Training loop
            train_loader = DataLoader(train_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            val_loader = DataLoader(val_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            loss_acc_history, val_loss_best, best_model = train(opt, train_loader, val_loader, model, optimizer, start_time)

            #End resource use logging and display
            memuse_b = tracemalloc.get_traced_memory()[1]
            time_seconds = time.time() - start_time
            print("Peak memory use: %.2fMeB" % (memuse_b / (1024 ** 2)))
            print(f"Total training time: {timedelta(seconds = time_seconds)}")
            tracemalloc.stop()

            #Testing with best model
            test_loader = DataLoader(test_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            test_loss_acc = validate(opt, test_loader, model)
            print("Test: ", test_loss_acc)


            #Saving model
            SAVE_ROOT = "./bp/models/MNIST"
            save_dir = f"{SAVE_ROOT}/backprop-{time.time()}"
            os.makedirs(save_dir, exist_ok = True)
            torch.save(best_model.state_dict(), f"{save_dir}/backprop-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}")

            #Metadata
            OmegaConf.save(opt, f"{save_dir}/config.yaml")
            training_info = loss_acc_history.copy()
            training_info["peak_memory_use"] = {
                "bytes": memuse_b,
                "kilobytes": memuse_b / 1024,
                "megabytes": memuse_b / (1024 ** 2),
                "gigabytes": memuse_b / (1024 ** 3)
            }
            training_info["time_elapsed_seconds"] = time_seconds
            with open(f"{save_dir}/training_history.json", "w", encoding = 'utf-8') as f:
                f.write(json.dumps(training_info, ensure_ascii = False, indent = 4))

    print(torch.cuda.memory_summary(device = "cuda"))


if __name__ == "__main__":
    mainfunc()