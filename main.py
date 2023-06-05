from mpp.mpptypes import LNGraph


# Set parameters
A = 0.25e8
N = 5
Q = 1000
MAX_ATTEMPTS = 6
USE_KNOWN_BALANCES = False


# Initialize graph
# NOTE: S and D must be up to date with channels_processed.pkl
#       (see preprocessing.py)
# TODO: Make this so that you don't have to update (S, D) every time
S = 4872
D = 16154
G = LNGraph(S, D, A, USE_KNOWN_BALANCES)


# Attempt payment
for attempt in range(MAX_ATTEMPTS):
    print(f"\033[4m Attempt {attempt + 1:>2}/{MAX_ATTEMPTS:>2} \033[0m")

    arcs = G.linearize(N, Q)
    flow = G.solve_mcf(arcs, A, Q)
    onions = G.flow_to_payment(flow)

    if not onions.bad_onions():
        print(f"Payment completed in {G.time:>.3f} seconds")
        exit(0)

    G.update_bounds(onions)
