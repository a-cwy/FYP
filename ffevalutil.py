import torch
import numpy as np

#Get samples
def getitem(sample, label):
    pos_sample, neg_sample, neutral_sample, class_label = generate_sample(sample, label)

    inputs = {
        "pos_images": pos_sample,
        "neg_images": neg_sample,
        "neutral_sample": neutral_sample,
    }
    labels = {"class_labels": class_label}

    return inputs, labels



#Generate positive sample
def get_pos_sample(sample, class_label):
    one_hot_label = torch.nn.functional.one_hot(
        torch.tensor(class_label), num_classes = 10
    )
    pos_sample = sample.clone()
    pos_sample[0, 0, 0, : 10] = one_hot_label
    return pos_sample



#Generate negative sample (random wrong label)
def get_neg_sample(sample, class_label):
    classes = list(range(10))
    classes.remove(class_label)
    wrong_class_label = np.random.choice(classes)
    one_hot_label = torch.nn.functional.one_hot(
        torch.tensor(wrong_class_label), num_classes = 10
    )
    neg_sample = sample.clone()
    neg_sample[0, 0, 0, :10] = one_hot_label
    return neg_sample



#Generate neutral sample (all labels uniform)
def get_neutral_sample(z):
    uniform_label = torch.ones(10) / 10
    z[0, 0, 0, :10] = uniform_label
    return z



#Generate samples from index on MNIST
def generate_sample(sample, class_label):
    # Get MNIST sample.
    pos_sample = get_pos_sample(sample, class_label)
    neg_sample = get_neg_sample(sample, class_label)
    neutral_sample = get_neutral_sample(sample)
    return pos_sample, neg_sample, neutral_sample, class_label