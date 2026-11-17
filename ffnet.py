import math
import torch
import torch.nn as nn
from ff import utils



#ReLU activation that passes gradient regardless of input value
class ReLU_pass_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()



#Forward-forward network
class FFnet(torch.nn.Module):
    def __init__(self, opt, width = 784, hidden_dim = 800):
        super(FFnet, self).__init__()

        self.opt = opt
        self.activation = ReLU_pass_grad()

        #Initialize model
        self.num_channels = [hidden_dim] * self.opt.model.num_layers
        self.model = nn.ModuleList([nn.Linear(width, self.num_channels[0])]) #MNIST 784, CIFAR10 3072
        for i in range(1, self.opt.model.num_layers):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        #Initialize losses
        self.ff_loss = nn.BCEWithLogitsLoss()
        self.running_means = [torch.zeros(self.num_channels[i], device = self.opt.device) + 0.5 for i in range(self.opt.model.num_layers)]

        #Linear classifier without bias for classification loss
        channels_for_classification_loss = sum(self.num_channels[-i] for i in range(self.opt.model.num_layers - 1))
        self.linear_classifier = nn.Sequential(nn.Linear(channels_for_classification_loss, 10, bias = False))
        self.classification_loss = nn.CrossEntropyLoss()

        self._init_weights()



    #Initialize weights of the model, normally distributed for hidden layers and zeroes for classifier
    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean = 0, std = (1 / math.sqrt(m.weight.shape[0])))
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)



    #Layer normalizer
    def _layer_norm(self, z, eps = 1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim = -1, keepdim = True)) + eps)



    #Update running means and calculate peer normalization loss
    def _calc_peer_normalization_loss(self, index, z):
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim = 0)
        self.running_means[index] = self.running_means[index].detach() * self.opt.model.momentum + mean_activity * (1 - self.opt.model.momentum)
        peer_loss = (torch.mean(self.running_means[index]) - self.running_means[index]) ** 2

        return torch.mean(peer_loss)



    #Calculate forward-forward loss
    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim = -1)

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z.shape[0]).item()

        return ff_loss, ff_accuracy



    #Forward pass
    def forward(self, inputs, labels):
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim = 0)
        posneg_labels = torch.zeros(z.shape[0], device = self.opt.device)
        posneg_labels[:self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        for index, layer in enumerate(self.model):
            #Layer and activation pass
            z = layer(z)
            z = self.activation.apply(z)

            #Layer normalizaiton
            z = z.detach()
            z = self._layer_norm(z)

        #Send to classification layer
        output = self.downstream_classification(inputs, labels)

        return output




    #Classification layer
    def downstream_classification(self, inputs, labels):
        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for index, layer in enumerate(self.model):
                z = layer(z)
                z = self.activation.apply(z)
                z = self._layer_norm(z)


                if index >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim = -1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim = -1, keepdim = True)[0]

        return output