from collections import defaultdict
from datetime import timedelta
import json
import os
import time
import tracemalloc
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, read_write
import utils



#Trains model
def train(opt, model, optimizer, start_time):
    #Log start time for tracking
    epoch_start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    best_model = model
    val_loss_best = [5.0, 0]
    loss_acc_history = {"loss": [], "accuracy": []}

    epoch = 0
    while(time.time() - start_time <= 600):
    # for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        # optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(train_results, scalar_outputs, num_steps_per_epoch)

        utils.print_results("train", time.time() - epoch_start_time, train_results, epoch)
        epoch_start_time = time.time()

        #Epoch validation
        if epoch % opt.training.val_index == 0 and opt.training.val_index != -1:
            val_result = validate_test(opt, model, "val", epoch = epoch)
            loss_acc_history["loss"].append(val_result["classification_loss"])
            loss_acc_history["accuracy"].append(val_result["classification_accuracy"])

            #Update val_loss best
            if val_result["classification_loss"] <= val_loss_best[0]:
                val_loss_best = (val_result["classification_loss"], epoch)


        # #Early stopper
        # if epoch > val_loss_best[1] + opt.training.early_stop_threshold:
        #     break

        epoch += 1

    return best_model, val_loss_best, loss_acc_history



#Validate/test model
def validate_test(opt, model, partition, epoch = None):
    #Log start time for tracking
    start_time = time.time()
    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    test_results = defaultdict(float)

    #Prevent updates
    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.downstream_classification(inputs, labels)
            test_results = utils.log_results(test_results, scalar_outputs, num_steps_per_epoch)

    utils.print_results(partition, time.time() - start_time, test_results, epoch = epoch)
    #Reset to training
    model.train()

    return test_results



#Hydra config
@hydra.main(config_path = ".", config_name = "config", version_base = None)
def mainfunc(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)

    #Resource use logging
    start_time = time.time()
    tracemalloc.start()

    param_list = {
        "learning_rate": [0.0005],
        "weight_decay": [0.0005]
    }
    # param_list = {
    #     "learning_rate": [5e-4, 1e-3, 5e-3],
    #     "weight_decay": [5e-4, 1e-3, 5e-3]
    # }

    for lr_index in range(len(param_list["learning_rate"])):
        for wd_index in range(len(param_list["weight_decay"])):
            with read_write(opt):
                opt.training.learning_rate = param_list["learning_rate"][lr_index]
                opt.training.weight_decay = param_list["weight_decay"][wd_index]
    
            model, optimizer = utils.get_model_optimizer(opt)

            print("Total trainable parameters: ", sum(p.numel() for p in model.parameters()))

            model, val_loss_best, loss_acc_history = train(opt, model, optimizer, start_time)
            validate_test(opt, model, "val")

            if opt.training.final_test:
                validate_test(opt, model, "test")

            #End resource use logging and display
            memuse_b = tracemalloc.get_traced_memory()[1]
            time_seconds = time.time() - start_time
            print("Peak memory use: %.2fMeB" % (memuse_b / (1024 ** 2)))
            print(f"Total training time: {timedelta(seconds = time_seconds)}")
            tracemalloc.stop()

            #Saving model
            SAVE_ROOT = "./ff/models/CIFAR10"
            save_dir = f"{SAVE_ROOT}/forwardforward-{time.time()}"
            os.makedirs(save_dir, exist_ok = True)
            torch.save(model.state_dict(), f"{save_dir}/forwardforward-frozen-valloss-{val_loss_best[0]}-epoch-{val_loss_best[1]}")

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