from pickle import dump, HIGHEST_PROTOCOL
from random import randint

from mpp.mpptypes import Channel

import matplotlib.pyplot as plt
import networkx as nx


# Generate dual-BA graph with 1024 nodes.
SRC, DST = None, None
while True:
    G = nx.dual_barabasi_albert_graph(10000, 2, 3, 0.5)
    p = nx.periphery(G)
    p = [node for node in p if G.degree[node] >= 3]

    # Pick the furthest pair of nodes
    dist = 0
    for i, s in enumerate(p):
        for d in p[i + 1 :]:
            temp = nx.shortest_path_length(G, s, d)
            if temp > dist:
                SRC, DST, dist = s, d, temp
    if dist >= 6:
        break
print(f"Generated graph with {len(G.edges())} edges")
print(f"Source: {SRC}\nDestination: {DST}")


# # Draw graph
# D = dict(G.degree)
# NODELIST = D.keys()
# NODESIZE = [v * 10 for v in D.values()]
# POS = nx.spring_layout(G)
# LABELS = {SRC: SRC, DST: DST}
# nx.draw(G, nodelist=NODELIST, node_size=NODESIZE, pos=POS, alpha=0.5, width=0.2)
# nx.draw_networkx_labels(G, pos=POS, labels=LABELS, font_weight="bold")

# # Highlight shortest path from source to destination
# path = nx.shortest_path(G, SRC, DST)
# path_sizes = [D[node] * 10 for node in path]
# path_edges = list(zip(path, path[1:]))
# nx.draw_networkx_nodes(G, pos=POS, nodelist=path, node_size=path_sizes, node_color="r")
# nx.draw_networkx_edges(G, pos=POS, edgelist=path_edges, edge_color="r")

# plt.switch_backend("TkAgg")
# plt.show()


# Construct channel graph and randomly assign:
#  - channel capacities from 1ksat to 10BTC
#  - balances from 0 to channel capacity
#  - proportional fee rates from 0 to 1%
#  - base fees from 0 to 0.01BTC
edges: dict[int, dict[int, Channel]] = {}
for i, j in G.edges():
    c = randint(1e3, 10 * 1e8)
    u = randint(0, c)
    r = randint(0, 0.01 * 1e6)
    b = randint(0, 0.01 * 1e8)
    if i in edges and j in edges[i]:
        print(f"WARNING: Encountered duplicate channel ({i}, {j})")
    else:
        edges[i] = {j: Channel(c, u, r, b, 0, c), **edges.get(i, {})}
        edges[j] = {i: Channel(c, c - u, r, b, 0, c), **edges.get(j, {})}


dump(edges, open("channels_generated.pkl", "wb"), HIGHEST_PROTOCOL)
