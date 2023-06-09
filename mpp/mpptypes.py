import json
import time

from pickle import load
from random import randint

from ortools.graph.python import min_cost_flow  # type: ignore


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Arc:
    def __init__(self, s: int, d: int, c: int, unit_cost: int) -> None:
        self.s = s
        self.d = d
        self.c = c
        self.unit_cost = unit_cost


class Channel:
    def __init__(self, c: int, u: int, r: int, b: int, lower: int, upper: int) -> None:
        self.c = c
        self.u = u
        self.r = r
        self.b = b
        self.lower = lower
        self.upper = upper


class Onion:
    # FIXME: failure_source_index == -1 to indicate success is messy
    def __init__(self, path: list[int], amount: int, failure_source_index: int) -> None:
        self.path = path
        self.amount = amount
        self.failure_source_index = (s := failure_source_index)
        self.failed_hop = (path[s], path[s + 1]) if s != -1 else None

    def hops(self) -> list[tuple[int, int]]:
        return [(s, d) for s, d in zip(self.path, self.path[1:])]

    def upstream_hops(self) -> list[tuple[int, int]]:
        if self.failure_source_index == -1:
            return self.hops()
        else:
            return self.hops()[: self.failure_source_index]

    def downstream_hops(self) -> list[tuple[int, int]]:
        if self.failure_source_index == -1:
            return []
        else:
            return self.hops()[self.failure_source_index :]


class Payment:
    def __init__(self, onions: list[Onion]) -> None:
        self.onions = onions

    def good_onions(self) -> list[Onion]:
        return [o for o in self.onions if o.failure_source_index == -1]

    def bad_onions(self) -> list[Onion]:
        return [o for o in self.onions if o.failure_source_index != -1]


class LNGraph:
    def __init__(self, S: int, D: int, A: int, use_known: bool) -> None:
        # Initialize attributes
        self.S = S
        self.D = D
        self.A = A
        self.R = A
        self.use_known = use_known
        self.cmax = 0
        self.time = 0

        # Read channels from file and set maximum capacity
        self.edges: dict[int, dict[int, Channel]] = load(
            open("channels_processed.pkl", "rb")
        )
        for s in self.edges:
            for d, edge in self.edges[s].items():
                self.cmax = max(self.cmax, edge.c)

                # Set lower and upper bounds, using known balances if available
                known = self.use_known and (s == self.S or d == self.S)
                edge.lower = edge.u if known else 0
                edge.upper = edge.u if known else edge.c

        # Run checks
        self.check_bounds()
        self.check_feasibility()

    def edge_probability(self, s: int, d: int, x: int) -> float:
        edge = self.edges[s][d]
        if self.use_known and (s == self.S or d == self.S):
            return x <= edge.u
        elif x < edge.lower:
            return 1
        elif x > edge.upper:
            return 0
        else:
            return (edge.upper + 1 - x) / (edge.upper + 1 - edge.lower)

    def edge_fee_in_sats(self, s: int, d: int, x: int) -> float:
        return self.edges[s][d].b / 1e3 + x * self.edges[s][d].r / 1e6

    def linearize(self, N: int, Q: int) -> list[Arc]:
        # Add linearized arcs up to upper bound
        arcs: list[Arc] = []
        for s in self.edges:
            for d, edge in self.edges[s].items():
                if self.use_known and (s == self.S or d == self.S):
                    arcs.append(Arc(s, d, int(edge.u / Q), 0))
                else:
                    for i in range(N):
                        c = int((edge.upper - edge.lower) / (N * Q))
                        unit_cost = (i + 1) * int(self.cmax / (edge.upper - edge.lower))
                        arcs.append(Arc(s, d, c, unit_cost))

        print(f"Linearized into {len(arcs)} arcs\n")
        return arcs

    def solve_mcf(self, arcs: list[Arc], Q: int) -> dict[int, dict[int, int]]:
        # Initialize MCF solver
        mcf = min_cost_flow.SimpleMinCostFlow()
        for arc in arcs:
            mcf.add_arc_with_capacity_and_unit_cost(arc.s, arc.d, arc.c, arc.unit_cost)
        for s in self.edges:
            mcf.set_node_supply(s, 0)
            for d in self.edges[s]:
                mcf.set_node_supply(d, 0)
        mcf.set_node_supply(self.S, int(self.R / Q))
        mcf.set_node_supply(self.D, -int(self.R / Q))

        # Solve MCF and update total time
        start = time.time()
        status = mcf.solve()
        end = time.time()
        self.time += end - start

        if status != mcf.OPTIMAL:
            print(f"ERROR: MCF solver returned status {status}")
            exit(1)

        # Collect linearized arcs into payment flow
        flow: dict[int, dict[int, int]] = {}
        for i in range(mcf.num_arcs()):
            if mcf.flow(i):
                s: int = mcf.tail(i)
                d: int = mcf.head(i)
                x: int = mcf.flow(i) * Q
                if s not in flow:
                    flow[s] = {d: x}
                else:
                    flow[s][d] = flow[s].get(d, 0) + x

        print(
            f"Solver finished in {end - start:4.3f} seconds\n"
            f"Minimum generalized quadratic cost: {mcf.optimal_cost():,}\n"
        )

        return flow

    # FIXME: Check the logic here. We are passing hops over one channel
    #       with aggregate amount greater than the channel balance.
    def flow_to_payment(self, flow: dict[int, dict[int, int]]) -> Payment:
        # Initialize payment amount, onion list, and residual graph
        amount = sum(flow[self.S].values())
        onions: list[Onion] = []
        residual_graph: dict[int, dict[int, int]] = {
            s: {d: self.edges[s][d].u for d in self.edges[s]} for s in self.edges
        }

        # Create onions until no more flow remains
        while amount > 0:
            # Initialize pointers, path, and onion amount
            curr: int = self.S
            path: list[int] = [curr]
            onion_amount: int = amount
            failure_source_index: int = -1

            # Find path and amount
            while curr != self.D:
                next = sorted(flow[curr].keys())[0]
                path.append(next)
                onion_amount = min(onion_amount, flow[curr][next])
                curr = next

            # Check for failed hops
            for i, (s, d) in enumerate(zip(path, path[1:])):
                if residual_graph[s][d] < onion_amount:
                    failure_source_index = i
                    break

            # Decrement remaining flow (and residual balance if no failed hops)
            for i, (s, d) in enumerate(zip(path, path[1:])):
                flow[s][d] -= onion_amount
                if flow[s][d] == 0:
                    del flow[s][d]
                if failure_source_index == -1:
                    residual_graph[s][d] -= onion_amount

            # Add onion to list
            onions.append(Onion(path, onion_amount, failure_source_index))
            amount -= onion_amount

        print(f"Broke flow into payment with {len(onions)} onions\n")
        return Payment(onions)

    def update_bounds(self, payment: Payment) -> None:
        # For good onions, reduce lower and upper bounds of each hop by onion amount
        for onion in payment.good_onions():
            for s, d in onion.hops():
                self.edges[s][d].lower = max(self.edges[s][d].lower - onion.amount, 0)
                self.edges[s][d].upper -= onion.amount
                self.edges[s][d].u -= onion.amount

                assert 0 <= self.edges[s][d].lower
                assert self.edges[s][d].lower <= self.edges[s][d].u
                assert self.edges[s][d].u <= self.edges[s][d].upper
                assert self.edges[s][d].upper <= self.edges[s][d].c

                self.edges[d][s].lower += onion.amount
                self.edges[d][s].upper = min(
                    self.edges[d][s].upper + onion.amount, self.edges[d][s].c
                )
                self.edges[d][s].u += onion.amount

                assert 0 <= self.edges[d][s].lower
                assert self.edges[d][s].lower <= self.edges[d][s].u
                assert self.edges[d][s].u <= self.edges[d][s].upper
                assert self.edges[d][s].upper <= self.edges[d][s].c

        # For each upstream hop in each bad onion, increase lower bound by onion amount
        for onion in payment.bad_onions():
            for s, d in onion.upstream_hops():
                if not self.use_known or (s != self.S and d != self.S):
                    self.edges[s][d].lower += onion.amount

        # For each failed hop in each bad onion, set upper bound to lower bound plus onion amount
        for onion in payment.bad_onions():
            s, d = onion.failed_hop
            if not self.use_known or (s != self.S and d != self.S):
                self.edges[s][d].upper = min(
                    self.edges[s][d].lower + onion.amount, self.edges[s][d].upper
                )

        # Update payment amount
        a = sum(o.amount for o in payment.good_onions())
        self.R -= (a := sum(o.amount for o in payment.good_onions()))
        print(f"Sent {a:<,} sats;", end=" ")
        print(f"{int(self.R):<,} ({100 * self.R / self.A:5.3f} %) remaining")

    def check_bounds(self) -> None:
        # Warn if any edge has balance out of bounds
        flag = False
        for s in self.edges:
            for d, edge in self.edges[s].items():
                assert 0 <= edge.lower <= edge.u <= edge.upper <= edge.c

        if not flag:
            print("Balance bounds check passed")
        print()

    def check_feasibility(self) -> None:
        # Check total outbound balance & capacity of source node
        s_out_edges = (edges := self.edges)[self.S]
        u_s_out_total = sum(s_out_edges[d].u for d in s_out_edges)
        c_s_out_total = sum(s_out_edges[d].c for d in s_out_edges)

        # Check total inbound balance & capacity of destination node
        d_in_edges = list(edges[s][self.D] for s in edges if self.D in edges[s])
        u_d_in_total = sum(edge.u for edge in d_in_edges)
        c_d_in_total = sum(edge.c for edge in d_in_edges)

        # Print total balances & capacities; warn if not feasible
        print(f"Source outbound balance / capacity:", end=f"{'':5}")
        print(f"{u_s_out_total:>13,} / {c_s_out_total:>13,}")
        print(f"Destination inbound balance / capacity:", end=" ")
        print(f"{u_d_in_total:>13,} / {c_d_in_total:>13,}")

        if u_s_out_total < self.A:
            print(f"WARNING: Source outbound balance < payment amount")
        if u_d_in_total < self.A:
            print(f"WARNING: Desination inbound balance < payment amount")
        if c_s_out_total < self.A:
            print(f"WARNING: Source outbound capacity < payment amount")
        if c_d_in_total < self.A:
            print(f"WARNING: Destination inbound capacity < payment amount")
        print()

    def print_flow(self, flow: dict[int, dict[int, int]]) -> None:
        # Print results
        print(
            f"{'Channel':^14}"
            f"{'':4}{'Flow':>13} / {'Capacity':>13}"
            f"{'':4}{'Bounds':^30}"
            f"{'':4}{'P_e(x_e)':^8}"
            f"{'':4}{'Fee':^13}"
            f"{'':4}{'Failed?':^7}"
        )

        total_probability = 1
        total_fee_in_sats = 0
        for s, outflows in sorted(flow.items()):
            for d, x in sorted(outflows.items()):
                total_probability *= (p_e := self.edge_probability(s, d, x))
                total_fee_in_sats += (f_e := self.edge_fee_in_sats(s, d, x))
                print(
                    f"({s:>5}, {d:>5})"
                    f"{'':4}{x:>13,} / {self.edges[s][d].c:>13,}"
                    f"{'':4}[{self.edges[s][d].lower:>13,} "
                    f" {self.edges[s][d].upper:>13,}]"
                    f"{'':4}{p_e:^8.3f}"
                    f"{'':4}{f_e:>13.3f}"
                    f"{'':4}{'[X]' if x > self.edges[s][d].u else '[ ]'}"
                )
        print()

        print(
            f"Total fee:          {int(total_fee_in_sats):7,} sats\n"
            f"Effective fee rate: {100 * total_fee_in_sats / self.A:7.3f} %\n"
            f"Total probability:  {100 * total_probability:7.3f} %\n"
        )

    def print_payment(self, payment: Payment) -> None:
        print(
            f"{'Onion':>5}{'':4}{'Amount':>13}{'':4}"
            f"{'Failed Hop':^14}{'':4}{'Path':^29}"
        )

        for i, onion in enumerate(payment.onions):
            print(f"{i:>5}{'':4}{onion.amount:>13,}{'':4}", end="")

            if onion.failure_source_index == -1:
                print(f"{'':14}{'':4}", end="[ ")
            else:
                s, d = onion.failed_hop
                print(f"({s:>5}, {d:>5}){'':4}", end="[ ")

            for s, _ in onion.upstream_hops():
                print(f"{s:>5}", end=" --> ")
            for s, _ in onion.downstream_hops():
                print(f"{s:>5}", end="     ")
            print(f"{onion.path[-1]:>5} ]")

        print()
