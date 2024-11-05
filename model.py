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
                * np.sqrt(1.0 / self.layers[layer_idx])
            )

    def init_deltas_D(self):
        self.deltas = []
        for layer_idx in range(len(self.layers) - 1):
            self.deltas.append(
                np.zeros((self.layers[layer_idx + 1], self.layers[layer_idx] + 1))
            )

    def init_D(self):
        self.D = []
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

    def init_gamma(self, loss):
        return self.activations[-1] * (1 - self.activations[-1]) * (loss)

    def compute_gammas(self, loss):
        gamma = self.init_gamma(loss)
        gammas = [gamma]
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            self.activations[layer_idx] = np.insert(self.activations[layer_idx], 0, 1)
            prev_gamma = gammas[-1]

            gamma = (
                (self.weights[layer_idx].T @ prev_gamma)
                * self.activations[layer_idx]
                * (1 - self.activations[layer_idx])
            )

            gammas.append(gamma)

            return gammas

    def backpropagation(self, X, y, loss):
        gamma = self.init_gamma(loss)

        for layer_idx in range(len(self.layers) - 1):
            self.activations[layer_idx] = np.insert(self.activations[layer_idx], 0, 1)
            print(self.activations[layer_idx].shape)

    def predict(self, X):
        self.forward(X)
        return self.activations[-1]

    def train(self, X, batch_size=1):
        start_time = time.time()
        for epoch in range(self.epochs):
            self.init_deltas_D()
            self.init_D()
            index = np.random.permutation(len(X))
            X = X[index]
            losses = []
            for i in range(batch_size):
                self.forward(X[i])
                loss = self.activations[-1] - X[i]
                losses.append(np.sum(loss**2))
                self.backpropagation(X[i], X[i], loss)
                # print(self.activations[-1], X[i])

            self.average_loss = np.mean(losses)
            self.average_losses.append(self.average_loss)
            print(f"Epoch: {epoch}, Loss: {self.average_loss}")
        end_time = time.time()
        print(
            f"Training Time: {end_time - start_time}, Loss: {self.average_loss}, Learning Rate: {self.alpha}, Regolation Term: {self.lambd}, Batch Size: {batch_size}"
        )
