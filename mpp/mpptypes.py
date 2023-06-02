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
        self.floor = 0
        self.ceil = c


class Onion:
    def __init__(self, path: list[int], amount: int, s_fail_index: int) -> None:
        self.path = path
        self.amount = amount
        self.s_fail_index = s_fail_index

    def hops(self) -> list[tuple[int, int]]:
        return [(s, d) for s, d in zip(self.path, self.path[1:])]

    def upstream_hops(self) -> list[tuple[int, int]]:
        return self.hops()[: self.s_fail_index]

    def failed_hop(self) -> tuple[int, int]:
        s = self.path[self.s_fail_index]
        d = self.path[self.s_fail_index + 1]
        return s, d


"""
    Channel           Flow /   Capacity	 Bounds            P_e(x_e) Fee   Failed?
(  783,  4997)	12,582,000 / 16,777,215	[   0  16,777,215]  0.250   16822  [X]

floor(  783,  4997):           0 ->  12,582,000	> u =     9,338,299 (!)

HTLC	Amount (sats) 	  Failed Hop  	   Path
   5	       446,000	              	[  783--> 4997-->  733-->15287]
   6	     3,750,000	(  733, 15287)	[  783--> 4997--> 2450-->  733-| 15287]
   7	     4,194,000	(  733, 15287)	[  783--> 4997--> 9742-->17963-->  733-| 15287]
   8	     4,192,000	              	[  783--> 4997-->10197--> 5602-->11281-->15287]
"""
