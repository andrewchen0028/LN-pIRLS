from pickle import load
from scipy.sparse import dok_array, csc_array
from scipy.sparse.linalg import lsqr, spsolve

from mpp.mpptypes import Channel

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


# NOTE: outputs are extremely small (~1e-10)
def uncertainty_coefficient(s: int, d: int) -> float:
    return -((2 / G[s][d].upper) ** 2) * np.log(1 / 2)


def fee_rate_decimal(s: int, d: int) -> float:
    return 1e6 * G[s][d].r


def base_fee_satoshi(s: int, d: int) -> float:
    return 1e3 * G[s][d].b


# Load channel graph and set parameters
G: dict[int, dict[int, Channel]] = load(open("channels_processed.pkl", "rb"))
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
A = np.array([[uncertainty_coefficient(*tuple(e)) for e in edges]]).T

# Proportional fee rates (m x 1)
b = np.array([[fee_rate_decimal(*tuple(e))] for e in edges]).T

# Node-edge incidence matrix (n x m)
C = dok_array((n, m), dtype=int)
for e, edge in enumerate(edges):
    C[tuple(edge)[0], e] = -1  # -1 if edge e exits node i
    C[tuple(edge)[1], e] = +1  # +1 if edge e enters node i

# Node demand vector (n x 1)
d = np.zeros((n, 1), dtype=int)
d[SRC] += int(AMT)
d[DST] -= int(AMT)

# NOTE: Need sparse / fast-Laplacian solvers to handle these systems
# Initial solution with unity weighting (min ||Cx - d||^2)
C = csc_array(C)
L = C @ C.transpose()

# print(is_diagonally_dominant(L.toarray()))
# print(is_graph_laplacian(L.toarray()))

import time

# Solving "Lx = b" with spsolve(L, b) takes ~30s

start = time.time()
print("Solving system...")
x = spsolve(L, d)
print(f"Elapsed: {time.time() - start:.2f}s")
print(f"Found x of length {len(x)}, with {np.count_nonzero(x)} non-zero entries")
f = C.T @ x
Cf = C @ f

print(Cf[0])

# Get average of Cf
avg = np.mean(Cf)
sd = np.std(Cf)
print(f"Average Cf: {avg:.2f}, std: {sd:.2f}")

# Count entries with Cf more than 6sd above mean
print(f"Entries more than 6sd above mean: {np.count_nonzero(Cf > avg + 6*sd)}")

# print(np.allclose(C @ f, d))


# print("Solving system...")
# x = lsqr(C, d)

# # Iterate
# for i in range(6):
#     W = A.T @ A + np.diag((b / np.abs(x)).T[0])
#     R = np.linalg.pinv(W)
#     x = R @ C.T @ np.linalg.pinv(C @ R @ C.T) @ d
