import numpy as np
from parameters import *


class NeuralNetwork:
    def __init__(self, layers: list = LAYERS, l: float = LAMBDA):
        self.weights = []
        self.activations = []
        self.layers = layers
        self.deltas = []
        self.D = []
        self.l = l

    def init_weights(self):
        for layer_idx in range(len(self.layers) - 1):

            self.weights.append(
                np.random.rand(self.layers[layer_idx + 1], self.layers[layer_idx] + 1)
            )
            self.deltas.append(
                np.zeros((self.layers[layer_idx + 1], self.layers[layer_idx] + 1))
            )
            self.D.append(
                np.zeros((self.layers[layer_idx + 1], self.layers[layer_idx] + 1))
            )

    def forward(self, x):
        self.activations.append(x)
        for layer_idx in range(len(self.layers) - 1):
            self.activations[layer_idx] = np.insert(self.activations[layer_idx], 0, 1)
            a = self.sigmoid(
                np.dot(self.weights[layer_idx], self.activations[layer_idx])
            )
            self.activations.append(a)
            self.activations[layer_idx] = self.activations[layer_idx][1:]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def init_gamma(self, y):
        return (
            self.activations[-1]
            * (1 - self.activations[-1])
            * (self.activations[-1] - y)
        )

    def backpropagation(self, X, y):
        gamma = self.init_gamma(y)
        gammas = [gamma]

        for layer_idx in range(len(self.layers) - 2, 0, -1):
            self.activations[layer_idx] = np.insert(self.activations[layer_idx], 0, 1)

            gamma = (
                self.weights[layer_idx].T.dot(gamma)
                * self.activations[layer_idx]
                * (1 - self.activations[layer_idx])
            )
            gammas.append(gamma)

            for row in range(self.layers[layer_idx + 1]):
                for col in range(self.layers[layer_idx] + 1):
                    self.deltas[layer_idx][row, col] = (
                        self.deltas[layer_idx][row, col]
                        + self.activations[layer_idx][col] * gammas[-2][row]
                    )
                    if col == 0:
                        self.D[layer_idx][row, col] = self.deltas[layer_idx][
                            row, col
                        ] / len(X)

                    else:
                        self.D[layer_idx][row, col] = (
                            self.deltas[layer_idx][row, col]
                            + self.l * self.weights[layer_idx][row, col]
                        ) / len(X)

        if layer_idx > 0:
            self.activations[layer_idx] = self.activations[layer_idx][1:]

    def train(self, X):
        for i in range(len(X)):
            self.forward(X[i])
            self.backpropagation(X[i], X[i])

            # not finished
