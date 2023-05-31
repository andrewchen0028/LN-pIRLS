import json, time

from ortools.graph.python import min_cost_flow  # type: ignore


def uniform_probability(s: int, d: int, a: int) -> float:
    """Computes uniform probability of a payment of amount `a` on a channel s-->d"""
    c = channel_graph[s][d]
    return float(c + 1 - a) / (c + 1)


def fee_msat(s: int, d: int, a: int) -> float:
    """Computes the fees of a payment of amount `a` on a channel s-->d"""
    base, rate = fee_graph[s][d]
    # Divide ppm by 1000 to be compatible with base_fee (measured in msat, not sats)
    return base + a * rate / 1000


# Amount (sats), linearization segments, HTLC size increment (sats)
A, N, Q = 50e6, 5, 1000

# Source and destination LNIDs
S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
channels = json.load(open("listchannels20220412_processed.json"))

# Construct LNID set and find max channel capacity
cmax = 0
lnids: set[str] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])
    cmax = max(cmax, e["c"])  # FIXME: this ignores subsequent channel recombination

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
    # Get source, destination, and capacity
    s, d, c = lnid_to_orid[e["s"]], lnid_to_orid[e["d"]], e["c"]

    # Put channels into channel graph, combining parallel channels
    channel_graph[s][d] = channel_graph[s][d] + c if d in channel_graph[s] else c
    fee_graph[s][d] = (e["b"], e["r"])  # FIXME: this ignores parallel channel fees
    for i in range(N):
        # FIXME: use optimal linearization
        arcs.append((s, d, int(c / (N * Q)), (i + 1) * int(cmax / c)))

# Invoke solver
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

# Create payment from all nonzero linearized flows
payment: dict[tuple[int, int], int] = {}
for orid in range(mcf.num_arcs()):
    if mcf.flow(orid):
        s: int = mcf.tail(orid)
        d: int = mcf.head(orid)
        f: int = mcf.flow(orid) * Q
        payment[(s, d)] = f if (s, d) not in payment else payment[(s, d)] + f

print(
    f"Paying {(A / 100.0e6):4.2f} BTC from {lnid_to_orid[S]}({S[:7]}...) to"
    f" {lnid_to_orid[D]}({D[:7]}...)\n"
    f"Solver finished in: {(end - start):4.3f} sec\n"
    f"Minimum approximated quadratic cost: {mcf.optimal_cost()}\n\n"
    f"{'Arc':^14}\t{'Flow':>10} /"
    f"{'Capacity':>10}\t{'P_e(x_e)':^10}\t{'Fee (sats)':^10}"
)

# Print all flows and compute total probability/fees
total_fee, total_probability = 0, 1
for (s, d), f in payment.items():
    channel_probability = uniform_probability(s, d, f)
    fee = fee_msat(s, d, f) / 1000
    total_fee += fee
    total_probability *= channel_probability
    print(
        f"({s:>5}, {d:>5})\t"
        f"{f:>10} / {channel_graph[s][d]:>10}\t"
        f"{channel_probability:^10.3f}\t"
        f"{fee:>10.3f}"
    )

print(f"\n{'Total probability: ':20}{total_probability * 100:>6.3f} %")
print(f"{'Total fee: ':20}{total_fee:6.3f} sats")
print(f"{'Fee rate: ':20}{(total_fee * 100.0 / A):>6.3f} %")
print(f"{'Arcs: ':20}{len(payment):>6}")
