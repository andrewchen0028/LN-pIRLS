from copy import copy
from json import load
from pickle import dump, HIGHEST_PROTOCOL
from random import randint

from mpp.mpptypes import Channel


# Load channels, drop unnecessary fields, and rename keys
channels: list[dict[str]] = load(open("listchannels20220412.json"))["channels"]
rename: dict[str, str] = {
    "source": "s",
    "destination": "d",
    "satoshis": "c",
    "base_fee_millisatoshi": "b",
    "fee_per_millionth": "r",
}
channels = [
    {new_key: channel[old_key] for old_key, new_key in rename.items()}
    for channel in channels
]

# Convert node IDs from strings to integers
lnids: set[str] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])
lnid_to_orid: dict[str, int] = {lnid: orid for orid, lnid in enumerate(lnids)}
for e in channels:
    e["s"] = lnid_to_orid[e["s"]]
    e["d"] = lnid_to_orid[e["d"]]

# Put channels into a dict {s: {d: { ... }}} so we can query reverse edges.
# Combine parallel channels, taking max base fee and fee rate.
edges: dict[int, dict[int, Channel]] = {}
for e in channels:
    s = e["s"]
    d = e["d"]
    c = e["c"]
    b = e["b"]
    r = e["r"]
    if s in edges and d in edges[s]:
        edges[s][d].c += c
        edges[s][d].b = max(edges[s][d].b, b)
        edges[s][d].r = max(edges[s][d].r, r)
    else:
        edges[s] = {d: Channel(c, 0, r, b, 0, c), **edges.get(s, {})}

# Make unidirectional channels symmetric.
for s in list(edges):
    for d in list(edges[s]):
        if d not in edges or s not in edges[d]:
            edges[d] = {s: copy(edges[s][d]), **edges.get(d, {})}

# Make channel capacities and fees symmetrical
for s in edges:
    for d in edges[s]:
        edges[s][d].c = max(edges[s][d].c, edges[d][s].c)
        edges[s][d].b = max(edges[s][d].b, edges[d][s].b)
        edges[s][d].r = max(edges[s][d].r, edges[d][s].r)

# Randomize ground truth balances
for s in edges:
    for d in edges[s]:
        edges[s][d].u = randint(0, edges[s][d].c)
        edges[d][s].u = edges[s][d].c - edges[s][d].u

# Run assertions
for s in edges:
    for d in edges[s]:
        # Capacities, base fees, and fee rates must be nonnegative
        assert edges[s][d].c >= 0
        assert edges[s][d].b >= 0
        assert edges[s][d].r >= 0

        # Channels must be symmetrical (except balances)
        assert edges[s][d].c == edges[d][s].c
        assert edges[s][d].b == edges[d][s].b
        assert edges[s][d].r == edges[d][s].r

        # Balances must between zero and channel capacity (inclusive)
        assert 0 <= edges[s][d].u <= edges[s][d].c

        # Balances must add to channel capacities
        assert edges[s][d].u + edges[d][s].u == edges[s][d].c


# Save channels and print source/destination ORIDs
dump(edges, open("channels_processed.pkl", "wb"), protocol=HIGHEST_PROTOCOL)
S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
print(f"Source ORID: {lnid_to_orid[S]}\nDestination ORID: {lnid_to_orid[D]}")
