import json

import numpy as np
import matplotlib.pyplot as plt


def mainfunc():
    memory_budgets = ["35mb", "50mb", "100mb"]
    algs = ["bp", "ff", "npdt"]
    datasets = ["MNIST", "CIFAR10"]

    values = {
        "bp": {
            "MNIST": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            },
            "CIFAR10": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            }
        },
        "ff": {
            "MNIST": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            },
            "CIFAR10": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            }
        },
        "npdt": {
            "MNIST": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            },
            "CIFAR10": {
                "35mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "50mb": {
                    "mean": 0.0,
                    "sd": 0.0
                },
                "100mb": {
                    "mean": 0.0,
                    "sd": 0.0
                }
            }
        },
    }

    for alg in algs:
        for dataset in datasets:
            for membug in memory_budgets:
                accs = []                

                for run in range(1, 4):
                    with open(f"./{alg}/models/{dataset}/10min-{membug}-{run}/training_history.json", "r", encoding = 'utf-8') as f:
                        data = json.loads(f.read())

                    accs.append(max(data["accuracy"]))

                values[alg][dataset][membug]["mean"] = np.mean(accs).item()
                values[alg][dataset][membug]["sd"] = np.std(accs).item()

    x = [35, 50, 100]

    for alg in algs:
        y_mean = []
        y_sd = []
        for membug in memory_budgets:
            y_mean.append(values[alg]["MNIST"][membug]["mean"])
            y_sd.append(values[alg]["MNIST"][membug]["sd"])

        plt.errorbar(x, y_mean, y_sd)
        plt.text(x[-1] + 1, y_mean[-1], alg, va = "center", fontsize = 10, weight = "bold")

    plt.xlim(30, 110)
    plt.ylim(0.8, 1)
    plt.xlabel("VRAM Budget (MB)")
    plt.ylabel("Accuracy")
    plt.title("MNIST")
    plt.savefig(f"./plots/membug/MNIST.png", format = "png")
    plt.show()

    for alg in algs:
        y_mean = []
        y_sd = []
        for membug in memory_budgets:
            y_mean.append(values[alg]["CIFAR10"][membug]["mean"])
            y_sd.append(values[alg]["CIFAR10"][membug]["sd"])

        plt.errorbar(x, y_mean, y_sd)
        plt.text(x[-1] + 1, y_mean[-1], alg, va = "center", fontsize = 10, weight = "bold")

    plt.xlim(30, 110)
    plt.ylim(0, 1)
    plt.xlabel("VRAM Budget (MB)")
    plt.ylabel("Accuracy")
    plt.title("CIFAR10")
    plt.savefig(f"./plots/membug/CIFAR10.png", format = "png")
    plt.show()


if __name__ == "__main__":
    mainfunc()