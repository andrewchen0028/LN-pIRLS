# %% Imports and setup
from copy import copy
from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert
from matplotlib import pyplot as plt
from random import randint
import networkx as nx
import numpy as np

jlPkg.add("Laplacians")
jl.seval("using Laplacians")
jl.seval("using SparseArrays")


# %% Function definitions
def draw_graph(G: nx.Graph, SRC: int, DST: int):
    # Draw graph
    D = dict(G.degree)
    NODELIST = D.keys()
    NODESIZE = [v * 10 for v in D.values()]
    POS = nx.spring_layout(G)
    LABELS = {SRC: SRC, DST: DST}
    nx.draw(G, nodelist=NODELIST, node_size=NODESIZE, pos=POS, width=0.2, alpha=0.5)
    nx.draw_networkx_labels(G, pos=POS, labels=LABELS, font_weight="bold")

    # Highlight shortest path from source to destination
    path = nx.shortest_path(G, SRC, DST)
    sizes = [D[node] * 10 for node in path]
    edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos=POS, nodelist=path, node_size=sizes, node_color="r")
    nx.draw_networkx_edges(G, pos=POS, edgelist=edges, edge_color="r")

    plt.switch_backend("TkAgg")
    plt.show()


def uncertainty_coefficient(G: nx.Graph, i: int, j: int) -> float:
    return -((2 / G[i][j]["c"]) ** 2) * np.log(1 / 2)


def fee_rate_decimal(G: nx.Graph, i: int, j: int) -> float:
    return G[i][j]["r"] / 1e6


def base_fee_satoshi(G: nx.Graph, i: int, j: int) -> float:
    return G[i][j]["b"] / 1e3


def J(G: nx.Graph, f: np.ndarray) -> float:
    j1 = sum(
        uncertainty_coefficient(G, i, j) * f[e] ** 2
        for e, (i, j) in enumerate(G.edges())
    )
    j2 = sum(
        fee_rate_decimal(G, i, j) * abs(f[e]) for e, (i, j) in enumerate(G.edges())
    )
    return j1 + j2


def np_matrix_to_jl(A: np.ndarray) -> jl.SparseMatrixCSC:
    A = A.tocoo()
    i_jl = jlconvert(T=jl.Vector[jl.Int64], x=A.row + 1)
    j_jl = jlconvert(T=jl.Vector[jl.Int64], x=A.col + 1)
    v_jl = jlconvert(T=jl.Vector[jl.Float64], x=A.data)
    return jl.SparseArrays.sparse(i_jl, j_jl, v_jl, A.shape[0], A.shape[1])


def print_iteration(
    i: int,
    w1: np.ndarray,
    w2: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    f: np.ndarray,
    x: np.ndarray,
) -> None:
    print(f"==== Iteration {i + 1} ====")
    print(f"\tmin\t\tmax\t\tavg")
    print(f"w1:\t{np.min(w1):+0.3e}\t{np.max(w1):+0.3e}\t{np.mean(w1):+0.3e}")
    print(f"w2:\t{np.min(w2):+0.3e}\t{np.max(w2):+0.3e}\t{np.mean(w2):+0.3e}")

    print(f"C@f[SRC]: {(C @ f)[SRC]:+0.3e}")
    print(f"C@f[DST]: {(C @ f)[DST]:+0.3e}")
    print(f"sum(C@f): {np.sum(C @ f):+0.3e}")
    print(
        f"J(f): {J(G, f):+0.3e}\t"
        f"||Ax - d||: {np.linalg.norm(jl.Laplacians.lap(A) * x - d):0.3e}\n"
    )


# %% Generate graph
# NOTE: For the symmetrical case, which balance do we use?
#       "A" matrix will be very different depending on local
#       or remote balance, and we don't know which one the
#       solver will use ahead of time.

# Generate directed dual-BA graph with 1024 nodes.
SRC: int
DST: int
AMT: int = 1e6
count = 0
while True:
    G = nx.dual_barabasi_albert_graph(1024, 2, 3, 0.5)
    p = nx.periphery(G)
    p = [node for node in p if G.degree[node] >= 3]

    # Pick the furthest pair of nodes
    dist = 0
    for i, s in enumerate(p):
        for d in p[i + 1 :]:
            smoothed = nx.shortest_path_length(G, s, d)
            if smoothed > dist:
                SRC, DST, dist = s, d, smoothed
    if dist >= 3:
        break
    if (count := count + 1) > 100:
        raise Exception("dist too large")


# Assign edge attributes
# NOTE: Consider making these dependent on node degree
for i, j in G.edges():
    G[i][j]["c"] = randint(1e3, 1e9)
    G[i][j]["u"] = randint(0, G[i][j]["c"])
    G[i][j]["r"] = randint(0, 0.01 * 1e6)
    G[i][j]["b"] = randint(0, 0.01 * 1e8)
    G[i][j]["e"] = copy(G[i][j]["c"])

C = nx.incidence_matrix(G, oriented=True)
n = G.number_of_nodes()
m = G.number_of_edges()

d = np.zeros(n)
d[SRC] += AMT
d[DST] -= AMT

print(f"Generated graph with {n} nodes and {m} edges")
print(f"Source: {SRC}\nDestination: {DST}")

# draw_graph(G, SRC, DST)


# %% Initialize solution and iterate
f = np.zeros(m)
for iteration in range(256):
    # Initialize edge weights
    w1 = np.zeros(m)  # uncertainty
    w2 = np.zeros(m)  # linear fees

    # # Assign edge weights without smoothing
    # for e, (i, j) in enumerate(G.edges()):
    #     w1[e] = 0 # uncertainty_coefficient(G, i, j)
    #     w2[e] = fee_rate_decimal(G, i, j) / (abs(f[e]) + 1)
    #     G[i][j]["w"] = (w1[e] + w2[e]) ** -1

    # Assign edge weights with smoothing
    smoothed = 0
    for e, (i, j) in enumerate(G.edges()):
        eps = G[i][j]["e"]
        if eps == 0:
            print("problem")
            exit(1)
        r = fee_rate_decimal(G, i, j)

        # TODO: count how many times we smooth w2
        # TODO: figure out why this is smoothing all edges at every iteration
        # NOTE: Seems like we'll run into divide-by-zero on w2?
        #       "eps" goes to zero, so does f[e] for many edges
        # NOTE: Issues may also be due to vastly different orders of magnitude of w1 and w2.
        if eps > r * abs(f[e]):
            smoothed += 1
        w1[e] = 0  # uncertainty_coefficient(G, i, j)
        w2[e] = r**2 / max(eps, r * abs(f[e]))

        G[i][j]["e"] /= 1.1
        G[i][j]["w"] = (w1[e] + w2[e]) ** -1

    # Get weighted adjacency matrix and solve for flow
    A = np_matrix_to_jl(nx.adjacency_matrix(G, weight="w"))
    jl.Laplacians.approxchol_lap
    jl.Laplacians.ApproxCholParams
    jl.Laplacians.approxchol_sddm
    jl.Laplacians.approxchol_lap2
    solver = jl.Laplacians.approxchol_lap2(A, verbose=False, tol=1e-6)
    x = solver(d)
    f = np.diag(np.reciprocal(w1 + w2)) @ C.T @ x
    print_iteration(iteration, w1, w2, A, C, f, x)
    print(f"Smoothed {smoothed} out of {m} edges")

print(f"Flow:{np.sort(f)[0:32]}")
print(f"Flow:{np.sort(f)[-32:]}")
