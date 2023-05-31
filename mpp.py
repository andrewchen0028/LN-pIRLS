import json, time

from ortools.graph.python import min_cost_flow  # type: ignore


# fmt: off
class Lnid(str): pass
class Orid(int): pass


class McfArc:
    def __init__(self, s: Orid, d: Orid, capacity: int, unit_cost: int) -> None:
        self.s, self.d = s, d
        self.capacity, self.unit_cost = capacity, unit_cost


class Channel:
    def __init__(self, capacity: int, prop_fee: int, base_fee: int,
                 balance: int, lower_bound: int, upper_bound: int) -> None:
        self.capacity, self.prop_fee, self.base_fee = capacity, prop_fee, base_fee
        self.balance, self.lower, self.upper = balance, lower_bound, upper_bound
# fmt: on


def uniform_probability(s: Orid, d: Orid, a: int) -> float:
    """Computes uniform probability of a payment of amount `a` on a channel s-->d"""
    return float(G[s][d].capacity + 1 - a) / (G[s][d].capacity + 1)


def fee_msat(s: Orid, d: Orid, a: int) -> float:
    """Computes the fees of a payment of amount `a` on a channel s-->d"""
    # Divide ppm by 1000 to be compatible with base_fee (measured in msat, not sats)
    return G[s][d].base_fee + a * G[s][d].prop_fee / 1000


# Amount (sats), linearization segments, HTLC size increment (sats)
A, N, Q = 50e6, 5, 1000

# Source and destination LNIDs
S = Lnid("03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df")
D = Lnid("021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d")
channels = json.load(open("listchannels20220412_processed.json"))

# Construct LNID set and find max channel capacity
cmax = 0
lnids: set[Lnid] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])
    cmax = max(cmax, e["c"])  # FIXME: this ignores subsequent channel recombination

# Map string LNIDs to integer ORIDs from [0, ..., n]
lnid_to_orid: dict[Lnid, Orid] = {lnid: Orid(i) for i, lnid in enumerate(lnids)}
orid_to_lnid: dict[Orid, Lnid] = {Orid(i): lnid for i, lnid in enumerate(lnids)}

# Initialize channel graph and linearized arcs
arcs: list[McfArc] = []
G: dict[Orid, dict[Orid, Channel]] = {lnid_to_orid[lnid]: {} for lnid in lnids}

for e in channels:
    # Put channels into channel graph, combining parallel channels
    # FIXME: use optimal linearization
    s, d, c = lnid_to_orid[e["s"]], lnid_to_orid[e["d"]], e["c"]
    G[s][d] = Channel(c, e["r"], e["b"], e["b"], 0, c)
    for i in range(N):
        arcs.append(McfArc(s, d, int(c / (N * Q)), (i + 1) * int(cmax / c)))

# Invoke solver
mcf = min_cost_flow.SimpleMinCostFlow()
for arc in arcs:
    mcf.add_arc_with_capacity_and_unit_cost(arc.s, arc.d, arc.capacity, arc.unit_cost)
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
payment: dict[tuple[Orid, Orid], int] = {}
for orid in range(mcf.num_arcs()):
    if mcf.flow(orid):
        s: Orid = mcf.tail(orid)
        d: Orid = mcf.head(orid)
        f: int = mcf.flow(orid) * Q
        payment[(s, d)] = payment[(s, d)] + f if (s, d) in payment else f

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
        f"{f:>10} / {G[s][d].capacity:>10}\t"
        f"{channel_probability:^10.3f}\t"
        f"{fee:>10.3f}"
    )

print(f"\n{'Total probability: ':20}{total_probability * 100:>6.3f} %")
print(f"{'Total fee: ':20}{total_fee:6.3f} sats")
print(f"{'Fee rate: ':20}{(total_fee * 100.0 / A):>6.3f} %")
print(f"{'Arcs: ':20}{len(payment):>6}")
