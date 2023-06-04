import json, random, statistics

# Load channels, drop unnecessary fields, and rename keys
channels = [
    {
        new_key: channel[old_key]
        for old_key, new_key in {
            "source": "s",
            "destination": "d",
            "satoshis": "c",
            "base_fee_millisatoshi": "b",
            "fee_per_millionth": "r",
        }.items()
    }
    for channel in json.load(open("listchannels20220412.json"))["channels"]
]
channels.sort(key=lambda e: e["s"] + e["d"])


# Drop channels with ridiculous base fees
print(f"Channel count: {len(channels)}")
base_fees = [int(e["b"]) for e in channels]
base_fees.sort()
base_fee_6sd = statistics.mean(base_fees) + 6 * int(statistics.stdev(base_fees))
to_drop = [e for e in channels if e["b"] > base_fee_6sd]
channels = [e for e in channels if e["b"] < 1e3 * base_fee_6sd]
print(
    f"Dropped {len(to_drop)} channels with base fees > 6sd above mean"
    f" (above {int(base_fee_6sd / 1e3)} sat)"
)

##############################################
### TODO(1): FIX NEGATIVE CHANNEL BALANCES ###
##############################################

# Randomize ground truth balances
u: dict[tuple[str, str], int] = {}
for e in channels:
    if (e["d"], e["s"]) in u:
        e["u"] = e["c"] - u[(e["d"], e["s"])]
    else:
        e["u"] = u[(e["s"], e["d"])] = random.randint(0, e["c"])


# Convert string LNIDs to integer ORIDs from [0, ..., n]
lnids: set[str] = set()
for e in channels:
    lnids.update([e["s"], e["d"]])

lnid_to_orid: dict[str, int] = {lnid: orid for orid, lnid in enumerate(lnids)}
orid_to_lnid: dict[int, str] = {orid: lnid for orid, lnid in enumerate(lnids)}

for e in channels:
    e["s"] = lnid_to_orid[e["s"]]
    e["d"] = lnid_to_orid[e["d"]]


# Save channels
json.dump(channels, open(f"channels_processed.json", "w"), indent=4)
S = "03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"
D = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"

print(f"Source ORID: {lnid_to_orid[S]}\nDestination ORID: {lnid_to_orid[D]}")
with open("file.json", "w") as fp:
    fp.write("[" + ",\n".join(json.dumps(i) for i in channels) + "]\n")
