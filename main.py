# import os

from mpp.mpptypes import LNGraph


# Set parameters
A = 0.5
N = 5
Q = 1000
MAX_ATTEMPTS = 6
USE_KNOWN_BALANCES = True

# TODO: Finish up debugging

# Initialize graph
# NOTE: S and D must be up to date with channels_processed.pkl
#       (see preprocessing.py)
S = 16284
D = 17450
G = LNGraph(S, D, A, USE_KNOWN_BALANCES)

arcs = G.linearize(N, Q)
flow = G.solve_mcf(arcs, A, Q)

# for attempt in range(MAX_ATTEMPTS):
#     os.system("clear")
#     print(f"Attempt {attempt + 1:>2}/{MAX_ATTEMPTS:>2}")


#     arcs = G.linearize(N, Q)
#     flow = G.solve_mcf(arcs, A, Q)
#     onions = G.flow_to_payment(flow)

#     G.print_latest()

#     if not onions.bad_onions():
#         print(f"Payment completed in {G.time:>5.3} seconds")
#         exit(0)

#     G.update_bounds(onions)
#     G.check_bounds()
#     input("Press enter to continue...")
