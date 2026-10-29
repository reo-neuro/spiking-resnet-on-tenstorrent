# Turn results of ttnn-replay-chain.py (S29) into the CSVs the breakdown dashboard (not included) reads: l1policies.csv (the step's
# device busy time and the number of ops that had to move L1 tensors out, per placement policy) and l1chain.csv (the
# busy time per op category, in breakdown-sew18.py's categories, per policy).
# Args: <out dir> <key>::<label>::<chain.json> ...   (in the order they are to be shown)
import csv
import json
import os
import statistics
import sys

ORDER = ["入力の準備", "conv（stem）", "conv", "BN", "LIF", "並べ替え（layer4 の LIF の前後）", "プーリングと全結合",
         "残差の足し算とその他"]


def categories(ops):
    """busy time (ms) per category, with breakdown-sew18.py's classify rules (ops in execution order)."""
    first_pool = next(i for i, o in enumerate(ops) if o["op"] == "ttnn.max_pool2d")
    mean_i = next(i for i, o in enumerate(ops) if o["op"] == "ttnn.mean")
    cats = dict.fromkeys(ORDER, 0.0)
    for i, o in enumerate(ops):
        if i < first_pool and o["op"] in ("ttnn.conv2d", "ttnn.concat"):
            cat = "conv（stem）"
        elif i < first_pool and o["op"] not in ("ttnn.generic", "ttnn.typecast"):
            cat = "入力の準備"
        elif o["op"] == "ttnn.conv2d":
            cat = "conv"
        elif o["op"] == "ttnn.generic":
            cat = "LIF"
        elif o["op"] in ("ttnn.typecast", "ttnn.max_pool2d", "ttnn.to_layout"):
            cat = "プーリングと全結合"
        elif o["op"] == "ttnn.add":
            cat = "残差の足し算とその他"
        elif o["op"] in ("ttnn.mean", "ttnn.linear") or i >= mean_i - 1:
            cat = "プーリングと全結合"
        elif o["op"] in ("ttnn.reshape", "ttnn.permute"):
            cat = "並べ替え（layer4 の LIF の前後）"
        else:
            cat = "残差の足し算とその他"
        cats[cat] += o["device_us"] / 1e3
    return cats


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
pol_rows, cat_rows = [], []
for order, spec in enumerate(sys.argv[2:]):
    key, label, path = spec.split("::")
    d = json.load(open(os.path.expanduser(path)))
    pol_rows.append({"order": order, "key": key, "label": label, "policy": d["policy"],
                     "ms": round(statistics.median(d["step_ms"]), 3),
                     "moves": sum(r["where"] == "MOVE" for r in d["ops"]),
                     "lif_mismatch": sum(r.get("lif_mismatch") or 0 for r in d["ops"])})
    for c, ms in categories(d["ops"]).items():
        cat_rows.append({"key": key, "category": c, "order": ORDER.index(c), "ms": round(ms, 3)})
for name, rows in (("l1policies.csv", pol_rows), ("l1chain.csv", cat_rows)):
    with open(os.path.join(out, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(name, len(rows), "rows")
for r in pol_rows:
    print(r["key"], r["ms"], r["moves"], r["lif_mismatch"])
