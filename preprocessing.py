import json, random, statistics

# Load channels, drop unnecessary fields, and rename fields
FILE = "listchannels20220412"
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
    for channel in json.load(open(f"{FILE}.json"))["channels"]
]

print(f"Channel count: {len(channels)}")
# Drop channels with ridiculous base fees
base_fees = [int(e["b"]) for e in channels]
base_fees.sort()
base_fee_6sd = int(sum(base_fees) / len(base_fees)) + 6 * int(
    statistics.stdev(base_fees)
)
to_drop = [e for e in channels if e["b"] > base_fee_6sd]
channels = [e for e in channels if e["b"] < 1e3 * base_fee_6sd]
print(
    f"Dropped {len(to_drop)} channels with base fees > 6sd above mean"
    f" (above {int(base_fee_6sd / 1e3)} sat)"
)


# Assign ground truth balances
u: dict[tuple[str, str], int] = {}
for e in channels:
    if (e["d"], e["s"]) in u:
        e["u"] = e["c"] - u[(e["d"], e["s"])]
    else:
        e["u"] = u[(e["s"], e["d"])] = random.randint(0, e["c"])

# Save channels
json.dump(channels, open(f"{FILE}_processed.json", "w"), indent=4)

with open("file.json", "w") as fp:
    fp.write("[" + ",\n".join(json.dumps(i) for i in channels) + "]\n")
