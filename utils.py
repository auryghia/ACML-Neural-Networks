import numpy as np


def data():
    matrix = np.zeros((8, 8), dtype=int)

    # Set a single '1' in each row at a different position
    for i in range(8):
        matrix[i, i] = 1

    print(matrix)
    return matrix
