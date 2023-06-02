import json, os, time

from ortools.graph.python import min_cost_flow  # type: ignore

from mpptypes import Arc, Channel, Onion


def edge_probability(s: int, d: int, x: int) -> float:
    if USE_KNOWN_BALANCES and s == lnid_to_orid[S] or d == lnid_to_orid[S]:
        return x <= G[s][d].u
    elif x < G[s][d].floor:
        return 1
    elif x > G[s][d].ceil:
        return 0
    else:
        return float((G[s][d].ceil + 1 - x) / (G[s][d].ceil + 1 - G[s][d].floor))


def edge_fee_sat(s: int, d: int, x: int) -> float:
    # NOTE: `b` is in msat, `r` is in ppm
    return G[s][d].b / 1e3 + x * G[s][d].r / 1e6


def print_flow(time, flow) -> None:
    print(
        f"Paying {(A / 100.0e6):5.3f} BTC from {lnid_to_orid[S]}({S[:7]}...) to"
        f" {lnid_to_orid[D]}({D[:7]}...)\n"
        f"Solver finished in {time:4.3f} seconds\n"
        f"Minimum generalized quadratic cost: {mcf.optimal_cost()}\n\n"
        f"{'Channel':^16}{'Flow':>14} /{'Capacity':>14}\t{'Bounds':^27}\t"
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
                f"{x:>14,} /{G[s][d].c:>14,}\t[{G[s][d].floor:>13,} {G[s][d].ceil:>13,}]\t"
                f"{p_e:^10.3f}\t{f_e:>7.0f}\t"
                f"{'  [X]  ' if x > G[s][d].u else '  [ ]  '}"
            )

    print(f"\n{'Total probability: ':20}{p_t * 100:>6.3f} %")
    print(f"{'Total fee: ':20}{f_t:6.0f} sats")
    print(f"{'Total fee rate: ':20}{(f_t * 100 / A):>6.3f} %")
    print(f"{'Arcs: ':20}{len(flow):>6}")


def check_flow(flow) -> bool:
    for s, out_flows in sorted(flow.items()):
        for d, x in sorted(out_flows.items()):
            if x > G[s][d].u:
                return False
    return True


def print_onions(onions: list[Onion]):
    print(f"\n{'HTLC':>4}\t{'Amount (sats)':<14}\t{'Failed Hop':^14}\t{'Path':^10}")
    for i, onion in enumerate(onions):
        if onion.s_fail_index == -1:
            print(f"{i + 1:>4}\t{onion.amount:>14,}\t{'':^14}\t[", end="")
        else:
            print(
                (
                    f"{i + 1:>4}\t{onion.amount:>14,}\t({onion.failed_hop()[0]:>5},"
                    f" {onion.failed_hop()[1]:5})\t["
                ),
                end="",
            )
        if onion.s_fail_index == -1:
            for orid in onion.path[:-1]:
                print(f"{orid:>5}", end="-->")
        else:
            for orid in onion.path[: onion.s_fail_index]:
                print(f"{orid:>5}", end="-->")
            print(f"{onion.path[onion.s_fail_index]:>5}", end="-| ")
            for orid in onion.path[onion.s_fail_index + 1 : -1]:
                print(f"{orid:>5}", end="   ")
        print(f"{lnid_to_orid[D]}]")
    print()


def flow_to_onions(
    flow: dict[int, dict[int, int]], balance_graph: dict[int, dict[int, int]]
):
    onions: list[Onion] = []
    remaining = A

    # Create onions until no more flow remains
    while remaining:
        # Initialize pointers, path, and amount
        curr, next = lnid_to_orid[S], None
        path: list[int] = []
        amt = remaining
        s_fail_index: int = -1

        # Find path and amount
        while curr != lnid_to_orid[D]:
            path.append(curr)
            next = sorted(flow[curr].keys())[0]
            amt = min(amt, flow[curr][next])
            curr = next
        path.append(curr)

        # Check for failed hops
        for i, (s, d) in enumerate(zip(path, path[1:])):
            if amt > balance_graph[s][d]:
                s_fail_index = i
                break

        # Decrement remaining flow and update balance graph if no failed hops
        for i, (s, d) in enumerate(zip(path, path[1:])):
            flow[s][d] -= amt
            if flow[s][d] == 0:
                del flow[s][d]
            if s_fail_index == -1:
                balance_graph[s][d] -= amt

        # Add onion to list
        onions.append(Onion(path, amt, s_fail_index))
        remaining -= amt

    return onions


# Set parameters
# Send A=50e6 with USE_KNOWN_BALANCES=False to observe two HTLCs on one channel,
# one successful and one failed
A, N, Q = int(50e6), 5, 1000
S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
USE_KNOWN_BALANCES = False
channels: list[dict[str, str | int]] = json.load(
    open("listchannels20220412_processed.json")
)
os.system("clear")


# Check outbound balance
u_out = 0
for e in channels:
    if e["s"] == S:
        u_out += e["u"]
if u_out < A:
    print(
        f"WARNING: insufficient outbound balance ({u_out / 1e8:6.3} <"
        f" {A / 1e8:6.3} BTC)"
    )
else:
    print(f"Outbound balance: {u_out / 1e8:6.3f} BTC")

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

    # Initialize channel if it doesn't exist, else combine with existing channel
    # NOTE: takes max of base fees and proportional fees for parallel channels
    if d not in G[s]:
        G[s][d] = Channel(e["c"], e["u"], e["r"], e["b"])
    else:
        G[s][d].c = G[s][d].c + e["c"]
        G[s][d].u = G[s][d].u + e["u"]
        G[s][d].r = max(G[s][d].r, e["r"])
        G[s][d].b = max(G[s][d].b, e["b"])

    # Initialize balance floors and ceilings
    if USE_KNOWN_BALANCES and e["s"] == S or e["d"] == S:
        G[s][d].floor = e["u"]
        G[s][d].ceil = e["u"]

for attempt in range(3):
    print(f"========== Attempt {attempt + 1:>2} ==========")
    arcs: list[Arc] = []
    for s in G.keys():
        for d in G[s].keys():
            # Add zero-cost arc if floor is known, then add linearized arcs up to ceil
            if G[s][d].floor:
                arcs.append(Arc(s, d, int(G[s][d].floor / Q), 0))
            n = min(N - 1, int((G[s][d].ceil - G[s][d].floor) / (N * Q)))
            for i in range(n):
                # fmt: off
                # TODO: confirm this cost logic is right; use optimal linearization
                arcs.append(Arc(s, d, int((G[s][d].ceil - G[s][d].floor) / (n * Q)),
                                (i + 1) * int(cmax / (G[s][d].ceil - G[s][d].floor))))
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
    print_flow(end - start, flow)
    if check_flow(flow):
        print("Payment successful!")
        break

    balance_graph = {
        s: {d: e.u for d, e in channels.items()} for s, channels in G.items()
    }
    onions: list[Onion] = flow_to_onions(flow, balance_graph)
    print_onions(onions)

    # TODO: Check bound-update logic below
    # Deduce balance bounds from this iteration's results alone
    floors: dict[int, dict[int, int]] = {}
    ceils: dict[int, dict[int, int]] = {}

    for onion in onions:
        for s, d in onion.hops():
            if s not in floors:
                floors[s] = {d: 0}
            else:
                floors[s][d] = 0

            if s not in ceils:
                ceils[s] = {d: G[s][d].c}
            else:
                ceils[s][d] = G[s][d].c

    for good_onion in [onion for onion in onions if onion.s_fail_index == -1]:
        for s, d in good_onion.hops():
            floors[s][d] += good_onion.amount

    for failed_onion in [onion for onion in onions if onion.s_fail_index != -1]:
        for s, d in failed_onion.upstream_hops():
            floors[s][d] += failed_onion.amount

    # TODO: Check why some failed hops' ceilings are not being reduced
    for failed_onion in [onion for onion in onions if onion.s_fail_index != -1]:
        s, d = failed_onion.failed_hop()
        ceils[s][d] = min(ceils[s][d], floors[s][d] + failed_onion.amount)
    print()

    # Update balance bounds where new bounds are stricter than existing
    # TODO: Fix floor-update logic (observed instance where floor > u)
    for s, d_dict in floors.items():
        for d, floor in d_dict.items():
            if floor > G[s][d].floor:
                print(
                    f"floor({s:>5}, {d:>5}): {G[s][d].floor:>11,} ->"
                    f" {floor:>11,}\t{'>' if floor > G[s][d].u else '<'} u ="
                    f" {G[s][d].u:>13,} {'(!)' if floor > G[s][d].u else ''}"
                )
            G[s][d].floor = max(G[s][d].floor, floor)
    print()

    for s, d_dict in ceils.items():
        for d, ceil in d_dict.items():
            if ceil < G[s][d].ceil:
                print(
                    f"ceil({s:>5}, {d:>5}): {G[s][d].ceil:>11,} ->"
                    f" {ceil:>11,}\t{'<' if ceil <= G[s][d].u else '>'} u ="
                    f" {G[s][d].u:>11,} {'(!)' if ceil < G[s][d].u else ''}"
                )
            G[s][d].ceil = min(G[s][d].ceil, ceil)
    print()
