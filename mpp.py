import json, time
from typing import TypeAlias

from ortools.graph.python import min_cost_flow  # type: ignore


def uniform_probability(a: int, s: int, d: int) -> float:
    """Computes uniform probability of a payment of amount `a` on a channel s-->d"""
    c = channel_graph[s][d]
    return float(c + 1 - a) / (c + 1)


def fee_msat(a: int, s: int, d: int) -> float:
    """Computes the fees of a payment of amount `a` on a channel s-->d"""
    base, rate = fee_graph[s][d]
    # Divide ppm by 1000 to be compatible with base_fee (measured in msat, not sats)
    return base + a * rate / 1000


A = 50e6  # Amount to send (sats)
N = 5  # Number of piecewise linear segments
Q = 1000  # HTLC size increment (sats)

S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
channels = json.load(open("listchannels20220412_processed.json"))

# Construct LNID set and find max channel capacity
cmax = 0
lnids: set[str] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])
    cmax = max(cmax, e["c"])
    # FIXME: max_cap ignores subsequent channel recombination

# Map string LNIDs to integer ORIDs from [0, ..., n]
lnid_to_orid: dict[str, int] = {lnid: orid for orid, lnid in enumerate(lnids)}
orid_to_lnid: dict[int, str] = {orid: lnid for orid, lnid in enumerate(lnids)}

# Construct channel/fee graphs and linearized arcs
# arcs:             stores linearized arcs      [(src, dest, capacity, unit_cost)]
# channel_graph:    stores channel capacities   {src: {dest: capacity}}
# fee_graph:        stores fees as a tuple      {src: {dest: (base_fee_msat, ppm)}}
arcs: list[tuple[int, int, int, int]] = []
channel_graph: dict[int, dict[int, int]] = {lnid_to_orid[lnid]: {} for lnid in lnids}
fee_graph: dict[int, dict[int, tuple[int, int]]] = {
    lnid_to_orid[lnid]: {} for lnid in lnids
}
for e in channels:
    s = lnid_to_orid[e["s"]]
    d = lnid_to_orid[e["d"]]
    c = e["c"]

    # Put channels into channel graph, combining parallel channels
    if d in channel_graph[s]:
        channel_graph[s][d] += c
    else:
        channel_graph[s][d] = c

    # FIXME: ignores parallel channel fees
    fee_graph[s][d] = (e["b"], e["r"])

    # FIXME: use optimal linearization
    unit_cost = int(cmax / c)
    for i in range(N):
        arcs.append((s, d, int(c / (N * Q)), (i + 1) * unit_cost))


mcf = min_cost_flow.SimpleMinCostFlow()

for arc in arcs:
    mcf.add_arc_with_capacity_and_unit_cost(arc[0], arc[1], arc[2], arc[3])

for orid in orid_to_lnid.keys():
    mcf.set_node_supply(orid, 0)
mcf.set_node_supply(lnid_to_orid[S], int(A / Q))
mcf.set_node_supply(lnid_to_orid[D], -int(A / Q))

start = time.time()
status = mcf.solve()
end = time.time()

if status != mcf.OPTIMAL:
    print(f"Error: min cost flow solver returned {status}")
    exit(1)

# Create total flow from all linearized edges with nonzero flows
total_flow = {}
for orid in range(mcf.num_arcs()):
    if mcf.flow(orid) == 0:
        continue

    s = mcf.tail(orid)
    d = mcf.head(orid)
    flow = mcf.flow(orid) * Q

    key = str(s) + ":" + str(d)
    if key in total_flow:
        total_flow[key] = (s, d, total_flow[key][2] + flow)
    else:
        total_flow[key] = (s, d, flow)

print(
    f"Paying {(A / 100.0e6):4.2f} BTC from {lnid_to_orid[S]}({S[:7]}...) to"
    f" {lnid_to_orid[D]}({D[:7]}...)\n"
    f"Solver finished in: {(end - start):4.3f} sec\n"
    f"Minimum approximated quadratic cost: {mcf.optimal_cost()}\n"
    f"{'Arc':^14}\t{'Flow':>10} /"
    f" {'Capacity':>10}\t{'P_e(x_e)':^10}\t{'Fee (sats)':^10}"
)

total_fee = 0
probability = 1

# Print all edges and compute total probability/fees
for orid, flow_value in total_flow.items():
    s, d, flow = flow_value
    channel_probability = uniform_probability(flow, s, d)
    fee = fee_msat(flow, s, d)
    total_fee += fee / 1000
    print(
        f"({s:>5}, {d:>5})\t"
        f"{flow:>10} / {channel_graph[s][d]:>10}\t"
        f"{channel_probability:^10.3f}\t"
        f"{fee / 1000:>10.3f}"
    )
    probability *= channel_probability

print(f"\nProbability of entire flow: {probability:6.4f}")
print(f"Total fee: {total_fee} sats\nEffective rate: {(total_fee * 100.0 / A):5.3f}")
print(f"Arcs included in payment flow: {len(total_flow)}")

# print("\nchannel_graph:")
# for orid, channels in list(channel_graph.items()):
#     print(f"{orid} {orid_to_lnid[orid]}\t{channels}")

# print("\nfee_graph:")
# for orid, v in list(fee_graph.items()):
#     print(f"{orid} {orid_to_lnid[orid]}\t{v}")
