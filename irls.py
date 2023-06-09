from pickle import load
from scipy.sparse import dok_array, diags
from scipy.sparse.linalg import cg

from mpp.mpptypes import Channel

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


def print_stats(x: np.ndarray) -> None:
    print(f"min\t\tmax\t\tmean\t\tmedian\t\tstd")
    print(f"{np.min(x):.2e}\t{np.max(x):.2e}\t", end="")
    print(f"{np.mean(x):.2e}\t{np.median(x):.2e}\t{np.std(x):.2e}")


# NOTE: outputs are extremely small (~1e-10)
# TODO: Try scaling coefficients by channel capacity
def uncertainty_coefficient(s: int, d: int) -> float:
    return -((2 / G[s][d].upper) ** 2) * np.log(1 / 2)


def fee_rate_decimal(s: int, d: int) -> float:
    return 1e-6 * G[s][d].r


def base_fee_satoshi(s: int, d: int) -> float:
    return 1e-3 * G[s][d].b


def J(A: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    J1 = (1 / 2) * f.T @ diags(np.square(A).flatten()) @ f
    J2 = b.flatten() @ np.abs(f)
    return (J1 + J2).flatten()[0]


def Q(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    f = W @ C.T @ x.reshape(n, 1)
    J1 = (1 / 2) * f.T @ diags(np.square(A).flatten()) @ f
    J2 = b.flatten() @ np.abs(f)
    print(f"{(J1 + J2).flatten()[0]:.3e}\t{np.min(f):.3e}\t{np.max(f):.3e}\t")
    print(f"\tResidual: {np.linalg.norm(L @ x - d)}")
    return J1 + J2


# Load channel graph and set parameters
G: dict[int, dict[int, Channel]] = load(open("channels_generated.pkl", "rb"))
SRC: int = 4872
DST: int = 16154
AMT: int = int(0.5e8)

# Count nodes
node_ids: set[int] = {s for s in G}
for s in G:
    node_ids.update(d for d in G[s])
n: int = len(node_ids)

# Count edges (both directions count as one)
edges: set[frozenset[int]] = {frozenset({s, d}) for s in G for d in G[s]}
m: int = len(edges)

# Uncertainty coefficients (m x m)
#   min         max         avg         med         std
#   1.414e-18   2.291e-06   3.913e-10   2.772e-12   1.251e-08
A = np.array([[uncertainty_coefficient(*tuple(e)) for e in edges]]).T

# Proportional fee rates (m x 1)
#   min         max         avg         med         std
#   0.00000     3758.28505  0.09531     0.00015     17.60490
b = np.array([[fee_rate_decimal(*tuple(e))] for e in edges]).T

# Node-edge incidence matrix (n x m)
C = dok_array((n, m), dtype=int)
for e, edge in enumerate(edges):
    C[tuple(edge)[0], e] = -1  # -1 if edge e exits node i
    C[tuple(edge)[1], e] = +1  # +1 if edge e enters node i
C = C.tocsc()

# Node demand vector (n x 1)
d = np.zeros((n, 1), dtype=int)
d[SRC] += int(AMT)
d[DST] -= int(AMT)

# Graph Laplacian (n x n)
L = C @ C.T

# Initial flow vector solution (m x 1)
x = cg(L, d)[0]
f = C.T @ x
print(f"Initial cost: {J(A, b, f):.3e}")
print(f"min|f|: {np.min(np.abs(f))}")

# Iterate
for i in range(1):
    # W: (m x m) diagonal weight matrix
    #       min         max         avg         med         std
    #   w1  2.001e-36   5.250e-12   1.568e-16   7.687e-24   2.688e-14
    #   w2  0.000       1.114e+01   6.105e-04   3.700e-07   5.838e-02
    # NOTE: A *= 1e13 to bring max(w1, w2) to same order of magnitude
    # NOTE: Padded w1 with 1 to avoid division by zero
    w1 = np.square(A) * 1e13 + 1
    w2 = (b / (np.abs(f) + 1)).T
    W = diags(np.reciprocal(w1 + w2).flatten())

    print_stats(w1)
    print_stats(w2)

    # Weighted graph Laplacian (m x m)
    print("Computing weighted Laplacian")
    L = C @ W @ C.T

    # Weighted node demand vector (n x 1)
    print("Computing x")
    print("J\t\tmin(f)\t\tmax(f)")
    result = cg(
        L,
        d,
        x,
        maxiter=256,
        callback=lambda x: Q(A, b, x),
    )
    if result[1] != 0:
        print("WARNING: CG solver did not converge")
        print(result[1])
    x = result[0]

    # Weighted flow vector solution (m x 1)
    f = W @ C.T @ x.reshape(n, 1)
    d_out = (C @ f).flatten()
    print(f"max(d): {np.sort(d_out)[-4:]}")
    print(f"min(d): {np.sort(d_out)[:4]}\n")
    print(f"d stats:")
    print_stats(d_out)
