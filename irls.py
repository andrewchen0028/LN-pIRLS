from pickle import load

import numpy as np

from mpp.mpptypes import Channel


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
node_ids: set[int] = set()
for s in G:
    node_ids.add(s)
    for d in G[s]:
        node_ids.add(d)
n: int = len(node_ids)

# Count edges (both directions count as one)
edges: set[frozenset[int]] = set()
for s in G:
    for d in G[s]:
        edges.add(frozenset({s, d}))
m: int = len(edges)

# Map edges "{s, d}" to indices "e"
edge_to_index: dict[frozenset[int, int], int] = {e: i for i, e in enumerate(edges)}
index_to_edge: dict[int, frozenset[int, int]] = {i: e for i, e in enumerate(edges)}

# Uncertainty coefficients (m x m)
A = np.array([[uncertainty_coefficient(s, d) for s, d in index_to_edge.values()]]).T

# Proportional fee rates (m x 1)
b = np.array([[fee_rate_decimal(s, d) for s, d in index_to_edge.values()]]).T

# Node-edge incidence matrix (n x m)
C = np.zeros((n, m), dtype=int)
for e, edge in index_to_edge.items():
    C[tuple(edge)[0]][e] = -1  # -1 if edge e exits node i
    C[tuple(edge)[1]][e] = +1  # +1 if edge e enters node i

# Node demand vector (n x 1)
d = np.zeros((n, 1), dtype=int)
d[SRC] += int(AMT)
d[DST] -= int(AMT)

"""
NOTE: Need sparse / fast-Laplacian solvers to handle these systems
# Initial solution with unity weighting
x = C.T @ np.linalg.pinv(C @ C.T) @ d

# Iterate
for i in range(6):
    W = A.T @ A + np.diag((b / np.abs(x)).T[0])
    R = np.linalg.pinv(W)
    x = R @ C.T @ np.linalg.pinv(C @ R @ C.T) @ d
"""
