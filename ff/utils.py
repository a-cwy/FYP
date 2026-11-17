from datetime import timedelta
import random
from omegaconf import OmegaConf
import torch
import torchvision
import numpy as np
import ffnet



#MNIST dataloader for FF
class FFMNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = self.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes



    #Get samples
    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(index)

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}

        return inputs, labels



    #Generate positive sample
    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample



    #Generate negative sample (random wrong label)
    def _get_neg_sample(self, sample, class_label):
        classes = list(range(self.num_classes))
        classes.remove(class_label)
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample



    #Generate neutral sample (all labels uniform)
    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z



    #Generate samples from index on MNIST
    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label



    #Collect partition
    def get_MNIST_partition(self, opt, partition):
        if partition in ["train", "val", "train_val"]:
            mnist = torchvision.datasets.CIFAR10(
                "./data/CIFAR10/",
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
        elif partition in ["test"]:
            mnist = torchvision.datasets.CIFAR10(
                "./data/CIFAR10/",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
        else:
            raise NotImplementedError

        if partition == "train":
            mnist = torch.utils.data.Subset(mnist, range(40000))
        elif partition == "val":
            mnist = torchvision.datasets.CIFAR10(
                "./data/CIFAR10/",
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            mnist = torch.utils.data.Subset(mnist, range(40000, 50000))

        return mnist



    def __len__(self):
        return len(self.mnist)



#Build model and optimizer
def get_model_optimizer(opt):
    model = ffnet.FFnet(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    #Create optimizer for model and downstream classifier
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.Adam(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
            },
        ]
    )
    return model, optimizer

    

#Compute accuracy
def get_accuracy(opt, output, target):
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size



#Return dataloader
def get_data(opt, partition):
    dataset = FFMNIST(opt, partition)

    #Fixed seed for reproducability
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size = opt.input.batch_size,
        drop_last = True,
        shuffle = True,
        worker_init_fn = seed_worker,
        generator = g,
        num_workers = 4,
        persistent_workers = True,
    )



#Worker function
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



#Send to cuda if possible
def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels



#Send to cuda
def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict



#Get learning rate cooldown
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr

    

#Updates learning rate of model
def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(opt, epoch, opt.training.learning_rate)
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(opt, epoch, opt.training.downstream_learning_rate)
    return optimizer



#Logging results
def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict



#Display given results
def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t\t"
        f"Time: {timedelta(seconds = iteration_time)}\n",
        end = "",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f}\n", end = "")
    print()
    print()



#Parse options
def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt