import networkx as nx
import numpy as np


np.set_printoptions(precision=2, suppress=True, floatmode="fixed")


def J(A, b, x):
    return (1 / 2) * x.T @ A.T @ A @ x + b.T @ np.abs(x)


G = nx.cycle_graph(6, create_using=nx.DiGraph())
G.add_edge(0, 3)

# Hardcode edge weights s.t. initial uncertainty solution is 0->3,
# and algorithm converges to 0->1->2->3 once fee rates are applied
for e in G.edges():
    G[e[0]][e[1]]["a"] = 100
    G[e[0]][e[1]]["b"] = 100
G[0][3]["a"] = 1
G[0][3]["b"] = 3000
G[0][1]["b"] = G[1][2]["b"] = G[2][3]["b"] = 1

# Get system matrices
A = np.diag(list(nx.get_edge_attributes(G, "a").values()))
b = np.array([list(nx.get_edge_attributes(G, "b").values())]).T
C = nx.incidence_matrix(G, oriented=True).todense()
d = np.array([[-10, 0, 0, 10, 0, 0]]).T
Q = np.linalg.inv(A.T @ A)

# Initial solution with unity weighting
x = C.T @ np.linalg.pinv(C @ C.T) @ d
print(f"\t\t\tx\t\t\tJ(x)")
print(f"{x.T}\t{J(A, b, x)}")

# Iterate
for i in range(6):
    W = A.T @ A + np.diag((b / np.abs(x)).T[0])
    R = np.linalg.pinv(W)
    x = R @ C.T @ np.linalg.pinv(C @ R @ C.T) @ d
    print(f"{x.T}\t{J(A, b, x)}")
