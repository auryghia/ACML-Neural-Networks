import numpy as np


def data():
    matrix = np.zeros((8, 8), dtype=int)

    for i in range(8):
        matrix[i, i] = 1

    return matrix
