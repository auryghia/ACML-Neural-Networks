# Assuming you have already imported the necessary libraries
import numpy as np
from parameters import *
from utils import data
from model import NeuralNetwork
import matplotlib.pyplot as plt

data = data()
nn = NeuralNetwork()
nn.init_weights()

nn.train(data)
losses = nn.average_losses

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights = nn.weights


def plot_weights(weights):
    for i, weight_matrix in enumerate(weights):
        plt.figure(figsize=(10, 5))

        # Calculate the min and max for the current weight matrix
        min_val, max_val = weight_matrix.min(), weight_matrix.max()
        print(f"Layer {i+1} - Min weight: {min_val}, Max weight: {max_val}")

        # If weights are all close to zero, set a manual vmin and vmax
        vmin, vmax = (min_val, max_val) if min_val != max_val else (-1, 1)

        sns.heatmap(
            weight_matrix,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            center=0,
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(f"Weights of Layer {i+1}")
        plt.xlabel("Next Layer Neurons")
        plt.ylabel("Current Layer Neurons")
        plt.show()


plot_weights(weights)


def plot_activations(activations):
    # Using the "plasma" color palette for better visibility
    for i, activation in enumerate(activations):
        plt.figure(figsize=(10, 5))
        sns.histplot(
            activation,
            bins=30,
            kde=True,
            color=sns.color_palette("plasma", as_cmap=True)(0.7),
        )
        plt.title(f"Activations of Layer {i+1}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.show()


plot_activations(activations)

plt.figure(figsize=(10, 6))
plt.plot(
    losses, label=f"Average Loss (Learning Rate: {ALPHA}, Alpha: {ALPHA})", color="blue"
)
plt.title("Average Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.grid()
plt.axhline(y=0, color="black", linestyle="--")
plt.show()
