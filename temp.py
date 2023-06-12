from juliacall import Main as jl

# import juliacall as jl

from juliacall import Main as jl
from juliacall import Pkg as jlPkg

# jlPkg.activate(
#     "./Laplacians.jl-1.4.0"
# )  # relative path to the folder where `MyPack/Project.toml` should be used here

# jlPkg.activate("./Laplacians.jl")
jl.seval("using Laplacians.jl")


import numpy as np


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
# jl = juliacall.newmodule("temp")
jl.println("Hello from Julia!")
x = jl.rand(range(10), 3, 5)
x._jl_display()
print(np.sum(x, axis=0))
