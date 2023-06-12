from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert
from matplotlib import pyplot as plt
from random import randint
from scipy.sparse import dok_array
import networkx as nx
import numpy as np


jlPkg.add("Laplacians")
jl.seval("using Laplacians")
jl.seval("using SparseArrays")


def draw_graph(G: nx.Graph):
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
    G = nx.dual_barabasi_albert_graph(64, 2, 3, 0.5)
    p = nx.periphery(G)
    p = [node for node in p if G.degree[node] >= 3]

    # Pick the furthest pair of nodes
    dist = 0
    for i, s in enumerate(p):
        for d in p[i + 1 :]:
            temp = nx.shortest_path_length(G, s, d)
            if temp > dist:
                SRC, DST, dist = s, d, temp
    if dist >= 2:
        break
    if (count := count + 1) > 100:
        raise Exception("dist too large")


# Assign edge attributes
#  - channel capacities from 1ksat to 10BTC
#  - balances from 0 to channel capacity
#  - proportional fee rates from 0 to 1%
#  - base fees from 0 to 0.01BTC
# NOTE: Consider making these dependent on node degree
for i, j in G.edges():
    G[i][j]["c"] = randint(1e3, 1e9)
    G[i][j]["u"] = randint(0, G[i][j]["c"])
    G[i][j]["r"] = randint(0, 0.01 * 1e6)
    G[i][j]["b"] = randint(0, 0.01 * 1e8)

print(f"Generated graph with {len(G.edges())} edges")
print(f"Source: {SRC}\nDestination: {DST}")

# draw_graph(G)


np.set_printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format})


# Initial solution with unity weighting
A = nx.adjacency_matrix(G).tocoo()
print(A.toarray()[0:4][0:4][0:4])
n = G.number_of_nodes()
d = np.zeros(n)
d[SRC] += AMT
d[DST] -= AMT
i_jl = jlconvert(T=jl.Vector[jl.Int64], x=A.row + 1)
j_jl = jlconvert(T=jl.Vector[jl.Int64], x=A.col + 1)
v_jl = jlconvert(T=jl.Vector[jl.Float64], x=A.data)
A = jl.SparseArrays.sparse(i_jl, j_jl, v_jl, n, n)

solver = jl.Laplacians.approxchol_lap(A, verbose=True, tol=1e-12)
x = solver(d)  # type juliacall.VectorValue
print(x)

residual_laplacians = jl.Laplacians.lap(A) * x - d
print(f"Residual norm: {np.linalg.norm(residual_laplacians):0.3e}")


# Initial solution: L x = d
# Iterate: L = C @ W @ C.T


# Assign edge weights
w1 = np.array([[uncertainty_coefficient(G, *tuple(edge)) for edge in G.edges()]]).T
w2 = np.array([[base_fee_satoshi(G, *tuple(edge))] for edge in G.edges()]).T
for e, (i, j) in enumerate(G.edges()):
    G[i][j]["w"] = w1[e] ** 2 + w2[e]
