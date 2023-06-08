import numpy as np


# Goal: make a Laplacian and confirm SDD


def is_diagonally_dominant(A: np.ndarray):
    diagonal = np.abs(A.diagonal())
    row_sums = np.sum(np.abs(A), axis=1)

    return np.all(diagonal >= row_sums - diagonal)


C = np.array(
    [
        [-1, 0, 0, 0, 0, 0, 1],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 1, -1, -1, 0, 0, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 1, 0, 1, -1],
    ]
)


print(C @ C.T)
print(is_diagonally_dominant(C @ C.T))
