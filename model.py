import numpy as np
from parameters import *
import time


class NeuralNetwork:

    def __init__(
        self,
        layers=LAYERS,
        alpha=ALPHA,
        epochs=EPOCHS,
        lambd=LAMBDA,
    ):
        self.layers = layers
        self.alpha = alpha
        self.epochs = epochs
        self.lambd = lambd
        self.weights = []
        self.activations = []
        self.deltas = []
        self.D = []
        self.average_loss = 0
        self.average_losses = []

    def init_weights(self):
        for layer_idx in range(len(self.layers) - 1):
            self.weights.append(
                np.random.rand(self.layers[layer_idx + 1], self.layers[layer_idx] + 1)
            )

    def init_deltas_D(self):
        for layer_idx in range(len(self.layers) - 1):
            self.deltas.append(
                np.zeros((self.layers[layer_idx + 1], self.layers[layer_idx] + 1))
            )

    def init_D(self):
        for layer_idx in range(len(self.layers) - 1):
            self.D.append(
                np.zeros((self.layers[layer_idx + 1], self.layers[layer_idx] + 1))
            )

    def forward(self, x):
        self.activations = [x]
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

        if layer_idx > 0:
            self.activations[layer_idx] = self.activations[layer_idx][1:]

    def modify_weights(self, batch_size):
        for layer_idx in range(len(self.layers) - 1):

            for row in range(self.layers[layer_idx + 1]):

                for col in range(self.layers[layer_idx] + 1):

                    if col == 0:
                        self.D[layer_idx][row, col] = (
                            self.deltas[layer_idx][row, col] / batch_size
                        )
                        self.weights[layer_idx][row, col] -= (
                            self.alpha * self.D[layer_idx][row, col]
                        )
                    else:
                        self.D[layer_idx][row, col] = (
                            self.deltas[layer_idx][row, col]
                            + self.lambd * self.weights[layer_idx][row, col]
                        ) / batch_size
                        self.weights[layer_idx][row, col] -= (
                            self.alpha * self.D[layer_idx][row, col]
                        )

    def train(self, X, batch_size=8):
        start_time = time.time()
        for epoch in range(self.epochs):
            self.init_deltas_D()
            self.init_D()
            total_loss = 0
            index = np.random.permutation(len(X))
            X = X[index]
            for i in range(batch_size):
                self.forward(X[i])
                self.backpropagation(X[i], X[i])
                loss = np.mean((self.activations[-1] - X[i]) ** 2)
                total_loss += loss

            self.modify_weights(batch_size)
            self.average_loss = total_loss / batch_size
            self.average_losses.append(self.average_loss)

        end_time = time.time()
        print(
            f"Training Time: {end_time - start_time}, Loss: {self.average_loss}, Learning Rate: {self.alpha}, Regolation Term: {self.lambd}, Batch Size: {batch_size}"
        )
