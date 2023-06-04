import os

from mpptypes import LNGraph


# # Set parameters
# A = 0.5
# N = 5
# Q = 1000
# MAX_ATTEMPTS = 6
# USE_KNOWN_BALANCES = True

# Set parameters
a = float(input("Enter amount to send in BTC [0.5]: "))
A = int(a * 1e8 if a else 0.5 * 1e8)
N = 5
Q = 1000

# Prompt user for max attempts
max_attempts = input("Enter max attempts [6]: ")
MAX_ATTEMPTS = int(max_attempts) if max_attempts else 6

# Toggles use of known balances for source-adjacent edges
user_input = input("Use known balances? [Y/n]: ").lower()
USE_KNOWN_BALANCES = True if user_input == "y" or not user_input else False

# Initialize graph
G = LNGraph("file.json", 7971, 1653, A, USE_KNOWN_BALANCES)

for attempt in range(MAX_ATTEMPTS):
    # os.system("clear")
    print(f"Attempt {attempt + 1:>2}/{MAX_ATTEMPTS:>2}")

    arcs = G.linearize(N, Q)
    flow = G.solve_mcf(arcs, A, Q)
    onions = G.flow_to_payment(flow)

    G.print_latest()

    if not onions.bad_onions():
        print(f"Payment completed in {G.time:>5.3} seconds")
        exit(0)

    G.update_bounds(onions)
    G.check_bounds()
    # input("Press enter to continue...")
