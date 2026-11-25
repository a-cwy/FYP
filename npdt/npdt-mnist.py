from datetime import timedelta
import json
import os
import random
import threading
import time
import tracemalloc

import numpy as np
import torch
from torch import nn, optim
from torch.multiprocessing import Process
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms


import npnet



#Globals
SEED = 42
LR = 0.001
T = 10
EPOCHS = 100
LABELDIM = 10



def validate(val_loader, cnn, label_embedder, blocks):
    alpha = torch.linspace(1.0, 0.1, T)
    total_loss = 0.0
    correct_predicitons = 0
    sample_count = 0

    for x, y in val_loader:
        current_batch_size = x.shape[0]

        x = x.to("cuda")
        x_features = cnn(x) 
        z_t = torch.randn(current_batch_size, LABELDIM).to("cuda")
        for t in reversed(range(T)):
            z_t = label_embedder(z_t)
            x_input = torch.cat([x_features, z_t], dim = 1)
            u_hat = blocks[t](x_input.detach().to("cuda"))
            z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)

        softmax = nn.functional.softmax(z_t)
        output = torch.argmax(softmax, dim = -1, keepdim = True).flatten()
        y_onehot = nn.functional.one_hot(y, num_classes = 10).float()

        CELoss = nn.CrossEntropyLoss()
        loss = CELoss(z_t.to("cuda"), y_onehot.to("cuda")).item()
        total_loss += loss

        correct_predicitons += torch.sum(output.to("cuda") == y.to("cuda")).item()
        sample_count += current_batch_size

    val_loss_acc = {
        "loss": total_loss / sample_count,
        "accuracy": correct_predicitons / sample_count
    }
    
    return val_loss_acc



#Trainer
def train(train_loader, val_loader, cnn, label_embedder, blocks, optimizers):
    epoch_loss = [0.0]
    batch_counter = 0

    loss_acc_history = {"loss": [], "accuracy": []}
    val_loss_best = [0.0, 0]
    best_model = [cnn, label_embedder, blocks]
    
    for epoch in range(EPOCHS):
        #Learning rate decay
        if epoch > (EPOCHS // 2):
            lr = LR * 2 * (1 + LR - epoch) / LR

        alpha = torch.linspace(1.0, 0.1, T)
    
        for x, y in train_loader:
            current_batch_size = x.shape[0]
            
            #Setting up noisy labels 
            u_y = torch.zeros(current_batch_size, LABELDIM).scatter_(1, y.unsqueeze(1), 1).to("cuda")
            z = [u_y]
            for t in range(1, T + 1):
                eps = torch.randn_like(u_y)
                z_t = torch.sqrt(alpha[t - 1]) * z[-1] + torch.sqrt(1 - alpha[t - 1]) * eps
                z.append(z_t)

            #Multithreaded training
            threads = []
            for index in range(T):
                threads.append(threading.Thread(target = train_block, args = (x, z[t - 1], u_y, cnn, label_embedder, blocks[index], optimizers[index], epoch_loss)))
                threads[index].start()

            for thread in threads:
                thread.join()

            batch_counter += 1

        #Epoch summary
        avg_loss = epoch_loss[0] / batch_counter
        print(f"Epoch {epoch}")
        print(f"Average loss {avg_loss}")

        #Validation
        val_loss_acc = validate(val_loader, cnn, label_embedder, blocks)
        print("Validation: ", val_loss_acc)
        loss_acc_history["loss"].append(val_loss_acc["loss"])
        loss_acc_history["accuracy"].append(val_loss_acc["accuracy"])

        # if val_loss_acc["accuracy"] > val_loss_best[0]:
        #     val_loss_best[0] = val_loss_acc["accuracy"]
        #     val_loss_best[1] = epoch

        #     save_dir = "./npdt/models/MNIST/final1"
        #     os.makedirs(f"{save_dir}/epoch-{val_loss_best[1]}", exist_ok = True)
        #     for index, block in enumerate(blocks):
        #         torch.save(block.state_dict(), f"{save_dir}/epoch-{val_loss_best[1]}/np-checkpoint-block-{index}")
        #     torch.save(cnn.state_dict(), f"{save_dir}/epoch-{val_loss_best[1]}/np-checkpoint-cnn")
    
    return loss_acc_history, val_loss_best, best_model


#Main
def mainfunc():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    #Dataset
    mnist = datasets.MNIST(root = "./data/MNIST/", download = True, train = True, transform = transforms.ToTensor())
    train_data = Subset(mnist, range(50000))
    val_data = Subset(mnist, range(50000, 60000))
    test_data = datasets.MNIST(root = "./data/MNIST/", download = True, train = False, transform = transforms.ToTensor())

    cnn = npnet.CNNmnist().to("cuda").eval()
    label_embedder = npnet.LabelEmbedder(LABELDIM).to("cuda").eval()
    blocks = nn.ModuleList([npnet.DenoiserBlock(LABELDIM).to("cuda") for _ in range(T)])
    optimizers = [optim.AdamW(params = block.parameters(), lr = LR) for block in blocks]

    train_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 128, shuffle = True)

    print(cnn)
    print(label_embedder)
    print(blocks)

    total_param = 0
    for block in blocks:
        total_param += sum(p.numel() for p in block.parameters() if p.requires_grad)
    print("Total trainable parameters: ", total_param)



    #Resource use logging
    start_time = time.time()
    tracemalloc.start()



    #Training Loop
    loss_acc_history, val_loss_best, best_model = train(train_loader, val_loader, cnn, label_embedder, blocks, optimizers)
        


    #End resource use logging and display
    memuse_b = tracemalloc.get_traced_memory()[1]
    time_seconds = time.time() - start_time
    print("Peak memory use: %.2fMeB" % (memuse_b / (1024 ** 2)))
    print(f"Total training time: {timedelta(seconds = time_seconds)}")
    tracemalloc.stop()
    
    #Testing with best model
    test_loader = DataLoader(test_data, batch_size = 128, shuffle = True, generator = torch.Generator(device = "cuda"))
    test_loss_acc = validate(test_loader, best_model[0], best_model[1], best_model[2])
    print("Test: ", test_loss_acc)
    
    
    #Saving model
    SAVE_ROOT = "./npdt/models/MNIST"
    save_dir = f"{SAVE_ROOT}/noprop-{time.time()}"
    os.makedirs(save_dir, exist_ok = True)
    torch.save(cnn.state_dict(), f"{save_dir}/noprop-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}-cnn")
    for index, block in enumerate(blocks):
        torch.save(block.state_dict(), f"{save_dir}/noprop-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}-block-{index}")
    
    #Metadata
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