import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix



#Main
def mainfunc():
    models = ["bp", "ff", "np"]
    eval_sets = ["MNIST", "CIFAR10"]
    modelnames = ["Backpropagation", "Forward-Forward", "NoProp"]

    for index, model in enumerate(models):
        for eval_set in eval_sets:

            with open(f"./evals/{model}/{eval_set}/res.json", "r", encoding = 'utf-8') as f:
                pred_data = json.loads(f.read())

            cm = ConfusionMatrixDisplay(confusion_matrix(pred_data["run1"]["true"], pred_data["run1"]["pred"]))

            cm.plot()
            plt.title(f"Confusion matrix ({modelnames[index]}, {eval_set})")
            plt.savefig(f"./plots/cm/{model}-{eval_set}-cm.png", format = "png")
            plt.close()

            losses = []
            accuracies = []

            for i in range(1, 4):
                path = f"./{model}/models/{eval_set}/final{i}/training_history.json"

                with open(path, "r", encoding = "utf-8") as f:
                    training_history = json.loads(f.read())

                losses.append(training_history["loss"])
                accuracies.append(training_history["accuracy"])

            loss_mean = np.mean(losses, axis = 0)
            loss_se = loss_mean / np.sqrt(loss_mean.size)
            loss_ci_boundary = (loss_mean - (1.96 * loss_se), loss_mean + (1.96 * loss_se))

            accuracy_mean = np.mean(accuracies, axis = 0)
            accuracy_se = accuracy_mean / np.sqrt(accuracy_mean.size)
            accuracy_ci_boundary = (accuracy_mean - (1.96 * accuracy_se), accuracy_mean + (1.96 * accuracy_se))

            plt.plot(loss_mean)
            plt.xlim((0, 100))
            plt.ylim((0, np.max(loss_ci_boundary) * 2))
            plt.fill_between(range(0, 100), loss_ci_boundary[0], loss_ci_boundary[1], alpha = 0.3)
            plt.title(f"Validation loss 95% CI ({modelnames[index]}, {eval_set})")
            plt.savefig(f"./plots/trainhist/loss/{model}-{eval_set}-loss.png", format = "png")
            plt.close()

            plt.plot(accuracy_mean)
            plt.xlim((0, 100))
            plt.ylim((0, 1))
            plt.fill_between(range(0, 100), accuracy_ci_boundary[0], accuracy_ci_boundary[1], alpha = 0.3)
            plt.title(f"Validation accuracy 95% CI ({modelnames[index]}, {eval_set})")
            plt.savefig(f"./plots/trainhist/acc/{model}-{eval_set}-acc.png", format = "png")
            plt.close()



if __name__ == "__main__":
    mainfunc()