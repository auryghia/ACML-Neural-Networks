# Assuming you have already imported the necessary libraries
import numpy as np
from parameters import *
from utils import data
from model import NeuralNetwork
import matplotlib.pyplot as plt

data = data()
nn = NeuralNetwork()
nn.init_weights()
nn.train(data, batch_size=8)

import seaborn as sns

weights = nn.weights
activations = nn.activations

import numpy as np
import matplotlib.pyplot as plt
import optuna


def objective(trial):
    lambd = trial.suggest_loguniform("lambda", 0.1, 1.0)
    alpha = trial.suggest_loguniform("alpha", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [1, 8])

    nn = NeuralNetwork(alpha=alpha, lambd=lambd)
    nn.init_weights()
    nn.train(data, batch_size=batch_size)

    return nn.average_losses[-1]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best loss: {study.best_value}")

best_params = study.best_params
best_nn = NeuralNetwork(alpha=best_params["alpha"], lambd=best_params["lambda"])
best_nn.init_weights()
best_nn.train(data, batch_size=best_params["batch_size"])

plt.figure(figsize=(12, 8))
plt.plot(
    best_nn.average_losses,
    color="blue",
    label=f'Best LAMBDA: {best_params["lambda"]}, ALPHA: {best_params["alpha"]}, BATCH SIZE: {best_params["batch_size"]}',
)
plt.title("Average Losses Over Epochs for Best Parameter Configuration", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Average Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
