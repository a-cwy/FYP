#Main
import json

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def mainfunc():
    models = ["bp", "ff", "np", "np-prl"]
    eval_sets = ["MNIST", "CIFAR10"]

    stats = {
        "bp": {
            "MNIST": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }
            },
            "CIFAR10": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }   
            }
        },
        "ff": {
            "MNIST": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }
            },
            "CIFAR10": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }   
            }
        },
        "np": {
            "MNIST": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }
            },
            "CIFAR10": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }   
            }
        },
        "np-prl": {
            "MNIST": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }
            },
            "CIFAR10": {
                "acc": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "f1": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "precision": {
                    "mean": 0.0,
                    "std": 0.0
                },
                "recall": {
                    "mean": 0.0,
                    "std": 0.0
                }   
            }
        }
    }

    for index, model in enumerate(models):
        for eval_set in eval_sets:
            
            with open(f"./evals/{model}/{eval_set}/res.json", "r", encoding = 'utf-8') as f:
                pred_data = json.loads(f.read())

                accuracies = []
                f1s = []
                precisions = []
                recalls = []

                for run in pred_data.values():
                    accuracies.append(accuracy_score(run["true"], run["pred"]))
                    f1s.append(f1_score(run["true"], run["pred"], average = "macro"))
                    precisions.append(precision_score(run["true"], run["pred"], average = "macro"))
                    recalls.append(recall_score(run["true"], run["pred"], average = "macro"))

                accuracies = [i * 100 for i in accuracies]
                f1s = [i * 100 for i in f1s]
                precisions = [i * 100 for i in precisions]
                recalls = [i * 100 for i in recalls]

                accuracy_mean = np.mean(accuracies)
                f1_mean = np.mean(f1s)
                precision_mean = np.mean(precisions)
                recall_mean = np.mean(recalls)

                accuracy_std = np.std(accuracies)
                f1_std = np.std(f1s)
                precision_std = np.std(precisions)
                recall_std = np.std(recalls)

                print()
                print(f"Accuracy: {accuracy_mean:.4f}%+-{accuracy_std:.4f}%")
                print(f"F1: {f1_mean:.4f}%+-{f1_std:.4f}%")
                print(f"Precision: {precision_mean:.4f}%+-{precision_std:.4f}%")
                print(f"Recall: {recall_mean:.4f}%+-{recall_std:.4f}%")

                stats[model][eval_set]["acc"]["mean"] = accuracy_mean.item()
                stats[model][eval_set]["acc"]["std"] = accuracy_std.item()
                    
                stats[model][eval_set]["f1"]["mean"] = f1_mean.item()
                stats[model][eval_set]["f1"]["std"] = f1_std.item()
                    
                stats[model][eval_set]["precision"]["mean"] = precision_mean.item()
                stats[model][eval_set]["precision"]["std"] = precision_std.item()
                    
                stats[model][eval_set]["recall"]["mean"] = recall_mean.item()
                stats[model][eval_set]["recall"]["std"] = recall_std.item()

    with open("./evals/stats.json", "w", encoding = "utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii = False, indent = 4))



if __name__ == "__main__":
    mainfunc()