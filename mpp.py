import json, time
from typing import cast

from ortools.graph.python import min_cost_flow  # type: ignore


class Arc:
    def __init__(self, s: int, d: int, c: int, unit_cost: int) -> None:
        self.s = s
        self.d = d
        self.c = c
        self.unit_cost = unit_cost


class Channel:
    def __init__(self, c: int, u: int, r: int, b: int) -> None:
        self.c = c
        self.u = u
        self.r = r
        self.b = b
        self.lower = 0
        self.upper = c


class Onion:
    def __init__(self, path: list[int], amount: int, s_fail: int | None = None) -> None:
        self.path = path
        self.amount = amount
        self.s_fail = s_fail


def edge_probability(s: int, d: int, x: int) -> float:
    if USE_KNOWN_BALANCES and s == lnid_to_orid[S] or d == lnid_to_orid[S]:
        return x <= G[s][d].u
    else:
        return float(G[s][d].c + 1 - x) / (G[s][d].c + 1)


def edge_fee_sat(s: int, d: int, x: int) -> float:
    # NOTE: `b` is in msat, `r` is in ppm
    return G[s][d].b / 1e3 + x * G[s][d].r / 1e6


def print_flow(start, end, flow) -> None:
    print(
        f"Paying {(A / 100.0e6):5.3f} BTC from {lnid_to_orid[S]}({S[:7]}...) to"
        f" {lnid_to_orid[D]}({D[:7]}...)\n"
        f"Solver finished in {end - start:4.3f} seconds\n"
        f"Minimum generalized quadratic cost: {mcf.optimal_cost()}\n\n"
        f"{'Channel':^16}{'Flow':>14} /{'Capacity':>14}\t"
        f"{'P_e(x_e)':^10}\t{'Fee':^7}\t{'Failed?':^7}\n"
    )

    f_t, p_t = 0, 1
    for s, out_flows in sorted(flow.items()):
        for d, x in sorted(out_flows.items()):
            p_e = edge_probability(s, d, x)
            f_e = edge_fee_sat(s, d, x)
            f_t += f_e
            p_t *= p_e
            print(
                f"({s:>5}, {d:>5})\t"
                f"{x:>14,} /{G[s][d].c:>14,}\t"
                f"{p_e:^10.3f}\t{f_e:>7.0f}\t"
                f"{'  [X]  ' if x > G[s][d].u else '  [ ]  '}"
            )

    print(f"\n{'Total probability: ':20}{p_t * 100:>6.3f} %")
    print(f"{'Total fee: ':20}{f_t:6.0f} sats")
    print(f"{'Total fee rate: ':20}{(f_t * 100 / A):>6.3f} %")
    print(f"{'Arcs: ':20}{len(flow):>6}")


def print_onions(onions):
    print(f"\n{'HTLC':>4}\t{'Amount (sats)':<14}\t{'Path':^10}")
    for i, onion in enumerate(onions):
        print(f"{i + 1:>4}\t{onion.amount:>14,}", end="\t[")
        upstream = True
        for orid in onion.path[:-1]:
            print(f"{orid:>5}", end="")
            # if orid == failed[0]:
            #     upstream = False
            #     print(f" X ", end="")
            # else:
            if True:
                print(f"{' - ' if upstream else ' - '}", end="")
        print(f"{lnid_to_orid[D]}]")


def flow_to_onions(flow):
    onions: list[Onion] = []
    remaining = A

    while remaining:
        # Initialize pointers, path, and amount
        curr, next = lnid_to_orid[S], None
        path: list[int] = []
        amt: int = remaining
        s_fail: int = -1

        # Find path from source to destination
        while curr != lnid_to_orid[D]:
            next = sorted(flow[curr].keys())[0]
            path.append(curr)
            amt = min(amt, flow[curr][next])
            curr = next

        # Update payment flow and check for failure
        for s, d in zip(path, path[1:]):
            flow[s][d] -= amt
            if flow[s][d] == 0:
                del flow[s][d]
            # if not s_fail and G[s][d].u < amt:
            #     s_fail = s
            #     d_fail = d

        # Add onion to list
        path.append(curr)
        onions.append(Onion(path, amt))
        remaining -= amt

    return onions


# Set parameters
A, N, Q = int(100e6), 5, 1000
S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
USE_KNOWN_BALANCES = True
channels = json.load(open("listchannels20220412_processed.json"))


# Check outbound liquidity
u_out = 0
for e in channels:
    if e["s"] == S:
        u_out += e["u"]
if u_out < A:
    print(f"WARNING: insufficient outbound ({u_out / 1e8:6.3} < {A / 1e8:6.3} BTC)")
else:
    print(f"Outbound liquidity: {u_out / 1e8:6.3f} BTC")


# Construct LNID set and find max channel capacity
cmax = 0
lnids: set[str] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])
    cmax = max(cmax, e["c"])  # FIXME: ignores subsequent channel recombination


# Map string LNIDs to integer ORIDs from [0, ..., n]
lnid_to_orid: dict[str, int] = {lnid: orid for orid, lnid in enumerate(lnids)}
orid_to_lnid: dict[int, str] = {orid: lnid for orid, lnid in enumerate(lnids)}


# Initialize linearized arc list and channel graph
arcs: list[Arc] = []
G: dict[int, dict[int, Channel]] = {lnid_to_orid[lnid]: {} for lnid in lnids}


# Put channels into channel graph, combining parallel channels
for e in channels:
    s = lnid_to_orid[e["s"]]
    d = lnid_to_orid[e["d"]]

    if d not in G[s]:
        G[s][d] = Channel(e["c"], e["u"], e["r"], e["b"])
    else:
        G[s][d].c = G[s][d].c + e["c"]
        G[s][d].u = G[s][d].u + e["u"]
        G[s][d].r = e["r"]
        G[s][d].b = e["b"]

    if USE_KNOWN_BALANCES and e["s"] == S or e["d"] == S:
        G[s][d].lower = e["u"]
        G[s][d].upper = e["u"]

    # Add zero-cost arc if lower bound is known, then add linearized arcs up to upper bound
    if G[s][d].lower:
        arcs.append(Arc(s, d, int(G[s][d].lower / Q), 0))
    n = min(N, int((G[s][d].upper - G[s][d].lower) / (N * Q)))
    for i in range(n):
        # fmt: off
        # TODO: confirm this cost logic is right; use optimal linearization
        arcs.append(Arc(s, d, int((G[s][d].upper - G[s][d].lower) / (n * Q)),
                        (i + 1) * int(cmax / (G[s][d].upper - G[s][d].lower))))
        # arcs.append(Arc(s, d, int(e["c"] / (N * Q)), (i + 1) * int(cmax / e["c"])))
        # fmt: on


# Invoke solver
mcf = min_cost_flow.SimpleMinCostFlow()
for arc in arcs:
    mcf.add_arc_with_capacity_and_unit_cost(arc.s, arc.d, arc.c, arc.unit_cost)
for orid in orid_to_lnid.keys():
    mcf.set_node_supply(orid, 0)
mcf.set_node_supply(lnid_to_orid[S], int(A / Q))
mcf.set_node_supply(lnid_to_orid[D], -int(A / Q))

start = time.time()
status = mcf.solve()
end = time.time()

if status != mcf.OPTIMAL:
    print(f"Error: MCF solver returned status {status}")
    exit(1)


# Combine linearized arcs into payment flow
flow: dict[int, dict[int, int]] = {}
for i in range(mcf.num_arcs()):
    if mcf.flow(i):
        s: int = mcf.tail(i)
        d: int = mcf.head(i)
        x: int = mcf.flow(i) * Q
        if s in flow:
            flow[s][d] = flow[s][d] + x if d in flow[s] else x
        else:
            flow[s] = {d: x}
print_flow(start, end, flow)


onions: list[Onion] = flow_to_onions(flow)
print_onions(onions)


# TODO: Update bounds for failed, upstream, and successful channels


# # Iterate through onions and update channel graph
# for onion in onions:
#     # Onion has a failed hop
#     if (s, d) := (onion.s_fail, onion.s_fail+1):
#         G[onion.s_fail][onion.s_fail + 1].upper = G


#     # If onion has a failed hop, update that one first
#     if onion.failed_hop:
#     if onion.failed_hop:
#         assert isinstance(onion.failed_hop, tuple)
#         failed_hop[0]
#     # Search onion for failed channel
#     if onion.failed_hop:
#         failed_hop[0]
#     for s, d in zip(onion.path, onion.path[1:]):
#         if G[s][d].u < onion.amount:
#             for u, v in zip(onion.path[0 : s - 1], onion.path[1 : d - 1]):
#                 G[u][v]. = onion.amount
#                 G[v][u].upper = G[v][u].capacity - onion.amount
#             G[s][d].upper = onion.amount
#             G[d][s].lower = G[d][s].capacity - onion.amount
#             failed_channels.append((s, d))
#             break
#     # Decrement channel balances
#     for s, d in zip(onion.path, onion.path[1:]):
#         G[s][d].balance -= onion.amount
#         G[d][s].balance += onion.amount
#     # Update lower and upper bounds
#     for s, d in zip(onion.path, onion.path[1:]):
#         G[s][d].lower = max(G[s][d].lower - onion.amount, 0)
#         G[s][d].upper = max(G[s][d].upper - onion.amount, 0)
#         G[d][s].lower = min(G[d][s].lower + onion.amount, G[d][s].capacity)
#         G[d][s].upper = min(G[d][s].upper + onion.amount, G[d][s].capacity)

# Print info
