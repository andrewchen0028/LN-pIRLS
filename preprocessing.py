import json
import random

# Load channels and drop unnecessary fields
FILE = "listchannels20220412"
# FILE = "testchannels"
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

# source                    s
# destination               d
# capacity                  c
# balance                   u
# balance upper bound       umax
# balance lower bound       umin
# proportional fee          r
# base fee                  b
