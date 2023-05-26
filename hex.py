import networkx as nx
import numpy as np


def J(A, b, x):
    return x.T @ A @ x + b.T @ np.abs(x)


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

C = nx.incidence_matrix(G, oriented=True).todense()
d = np.array([[-10, 0, 0, 10, 0, 0]]).T

A = np.diag(list(nx.get_edge_attributes(G, "a").values()))
b = np.array([list(nx.get_edge_attributes(G, "b").values())]).T
Q = np.linalg.inv(A.T @ A)

x = Q @ C.T @ np.linalg.pinv(C @ Q @ C.T) @ d
print(x.T.round())
print(f"J(x0) = {J(A, b, x)}")

for i in range(3):
    W = np.diag(np.array(A @ A @ x * x + b * np.abs(x)).T[0])
    R = np.linalg.inv(W.T @ W)
    x = R @ C.T @ np.linalg.pinv(C @ R @ C.T) @ d
    print(x.T.round())
    print(f"J(x{i}) = {J(A, b, x)}")
