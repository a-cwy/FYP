from datetime import timedelta
import random
import json
import os
import threading
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
from npnet import DenoiserBlock, CNN



#Multithreaded training
def train_block(losses_arr, block, optimizer, x_features, z, u_y):
    u_hat = block(x_features.detach().to("cuda"), z.detach().to("cuda"))
    loss = torch.mean((u_hat - u_y) ** 2)
    losses_arr.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




#Trainer
def train(opt, train_loader, val_loader, cnn, blocks, optimizers):
    alpha = torch.linspace(1.0, 0.1, opt.model.T)
    loss_acc_history = {"loss": [], "accuracy": []}
    val_loss_best = [0.0, 0]
    best_model = [cnn, blocks]

    for epoch in range(opt.training.epochs):
        epoch_loss = 0.0
        batch_count = 0

        #Learning rate decay
        lr = opt.training.learning_rate
        if epoch > (opt.training.epochs // 2):
            lr = opt.training.learning_rate * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs

        for optimizer in optimizers:
            optimizer.param_groups[0]["lr"] = lr

        for x, y in train_loader:
            current_batch_size = x.shape[0]
            u_y = torch.zeros(current_batch_size, opt.model.label_dim).scatter_(1, y.unsqueeze(1), 1)

            #Setting up noisy labels 
            z = [u_y]
            for t in range(1, opt.model.T + 1):
                eps = torch.randn_like(u_y)
                z_t = torch.sqrt(alpha[t - 1]) * z[-1] + torch.sqrt(1 - alpha[t - 1]) * eps
                z.append(z_t)

            #Training denoiser blocks
            x = x.to("cuda")
            x_features = cnn(x)
            losses = []

            threads = []
            for t in range(opt.model.T):
                threads.append(threading.Thread(target = train_block, args = (losses, blocks[t], optimizers[t], x_features, z[t - 1], u_y)))
                threads[t].start()

            for thread in threads:
                thread.join()

            total_loss = sum(losses)

            epoch_loss += total_loss.item()
            batch_count += 1

        #Epoch summary
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch}")
        print(f"Average loss {avg_loss}")

        #Validation
        val_loss_acc = validate(opt, cnn, val_loader, blocks)
        print("Validation: ", val_loss_acc)
        loss_acc_history["loss"].append(val_loss_acc["loss"])
        loss_acc_history["accuracy"].append(val_loss_acc["accuracy"])

        if val_loss_acc["accuracy"] > val_loss_best[0]:
            val_loss_best[0] = val_loss_acc["accuracy"]
            val_loss_best[1] = epoch

            save_dir = "./np/models/CIFAR10/final3-prl"
            os.makedirs(f"{save_dir}/epoch-{val_loss_best[1]}", exist_ok = True)
            for index, block in enumerate(blocks):
                torch.save(block.state_dict(), f"{save_dir}/epoch-{val_loss_best[1]}/np-checkpoint-block-{index}")
            torch.save(cnn.state_dict(), f"{save_dir}/epoch-{val_loss_best[1]}/np-checkpoint-cnn")

        #Early stopper
        if epoch > val_loss_best[1] + opt.training.early_stop_threshold:
            break

    return loss_acc_history, val_loss_best, best_model



#Validator
def validate(opt, cnn, val_loader, blocks):
    alpha = torch.linspace(1.0, 0.1, opt.model.T)
    total_loss = 0.0
    correct_predicitons = 0
    sample_count = 0

    for x, y in val_loader:
        current_batch_size = x.shape[0]
        z_t = torch.randn(current_batch_size, opt.model.label_dim)

        x = x.to("cuda")
        x_features = cnn(x) 
        for t in reversed(range(opt.model.T)):
            u_hat = blocks[t](x_features, z_t)
            z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)

        softmax = nn.functional.softmax(z_t)
        output = torch.argmax(softmax, dim = -1, keepdim = True).flatten()
        y_onehot = nn.functional.one_hot(y, num_classes = 10).float()

        CELoss = nn.CrossEntropyLoss()
        loss = CELoss(z_t, y_onehot).item()
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

    #Dataset
    cifar10 = datasets.CIFAR10(root = "./data/CIFAR10/", download = True, train = True, transform = transforms.ToTensor())
    train_data = Subset(cifar10, range(40000))
    val_data = Subset(cifar10, range(40000, 50000))
    test_data = datasets.CIFAR10(root = "./data/CIFAR10/", download = True, train = False, transform = transforms.ToTensor())

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

            cnn = CNN()
            blocks = nn.ModuleList([DenoiserBlock(label_dim = opt.model.label_dim, hidden_size = 320).to("cuda") for _ in range(opt.model.T)])
            optimizers = [optim.AdamW(block.parameters(), lr = opt.training.learning_rate) for block in blocks]

            print(cnn)
            print(blocks)

            total_param = 0
            for block in blocks:
                total_param += sum(p.numel() for p in block.parameters() if p.requires_grad)
            print("Total trainable parameters: ", total_param)

            #Resource use logging
            start_time = time.time()
            tracemalloc.start()



            #Training loop
            train_loader = DataLoader(train_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            val_loader = DataLoader(val_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            loss_acc_history, val_loss_best, best_model = train(opt, train_loader, val_loader, cnn, blocks, optimizers)



            #End resource use logging and display
            memuse_b = tracemalloc.get_traced_memory()[1]
            time_seconds = time.time() - start_time
            print("Peak memory use: %.2fMeB" % (memuse_b / (1024 ** 2)))
            print(f"Total training time: {timedelta(seconds = time_seconds)}")
            tracemalloc.stop()

            #Testing with best model
            test_loader = DataLoader(test_data, batch_size = opt.input.batch_size, shuffle = True, generator = torch.Generator(device = "cuda"))
            test_loss_acc = validate(opt, best_model[0], test_loader, best_model[1])
            print("Test: ", test_loss_acc)


            #Saving model
            SAVE_ROOT = "./np/models/CIFAR10"
            save_dir = f"{SAVE_ROOT}/noprop-{time.time()}"
            os.makedirs(save_dir, exist_ok = True)
            torch.save(cnn.state_dict(), f"{save_dir}/noprop-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}-cnn")
            for index, block in enumerate(blocks):
                torch.save(block.state_dict(), f"{save_dir}/noprop-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}-block-{index}")

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


if __name__ == "__main__":
    mainfunc()