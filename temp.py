from scipy.sparse.linalg import cg

import numpy as np


def is_diagonally_dominant(A: np.ndarray):
    diagonal = np.abs(A.diagonal())
    row_sums = np.sum(np.abs(A), axis=1)

    return np.all(diagonal >= row_sums - diagonal)


def is_graph_laplacian(A: np.ndarray):
    # Check if the input is a 2D square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print("Matrix is not square")
        return False

    # Check if the matrix is symmetric
    if not np.allclose(A, A.T):
        print("Matrix is not symmetric")
        return False

    # Check if the diagonal elements are non-negative
    if np.any(np.diag(A) < 0):
        print("Matrix has negative diagonal elements")
        return False

    # Check if the matrix has zero row sums
    if np.any(np.sum(A, axis=1) != 0):
        print("Matrix has non-zero row sums")
        return False

    print("Matrix is a graph Laplacian")
    return True


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

d = np.array([[-1, 0, 0, 1, 0, 0]]).T


L = C @ C.transpose()
is_graph_laplacian(L)


x = cg(L, d)


if list(x)[1] == 0:
    f = C.T @ list(x)[0]
    print(f)
else:
    print("CG failed")
