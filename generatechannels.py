import json

# fmt: off
# source, destination, capacity, balance, base fee, proportional fee
channels = [
    ["s", "a", 2, 2, 0, 0],
    ["a", "b", 2, 2, 0, 0],
    ["b", "d", 4, 1, 0, 0],
    ["s", "x", 1, 0, 0, 0],
    ["x", "y", 7, 5, 0, 0],
    ["y", "d", 4, 3, 0, 0],
    ["b", "x", 9, 4, 0, 0],
]
# fmt: on

channels = channels + [[e[1], e[0], e[2], e[2] - e[3], e[4], e[5]] for e in channels]

json.dump(
    {
        "channels": [
            {
                "s": c[0],
                "d": c[1],
                "c": c[2],
                "u": c[3],
                "b": c[4],
                "r": c[5],
            }
            for c in channels
        ]
    },
    open("testchannels.json", "w"),
)
