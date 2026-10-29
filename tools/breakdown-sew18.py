# Combine the per-op device times of SEW-ResNet18 on p150a (ttnn-replay-ir.py) and the per-layer GPU times on the A30
# (a30-sew18-profile.py) into one JSON for the breakdown page: totals by category, conv and LIF layer by layer (p150a
# ops matched to the model's layer names by shape, in execution order), and the measured step times. With a CSV
# directory, also writes the files the breakdown dashboard (not included) reads (S22 to S24, S27, S28, S34): steps, categories, ops,
# conv_layers, lif_layers, history, bounds, bound_params. The measured step times are not in the replay files; they are the constants in STEPS and
# HISTORY below (from the xla-sew18.py logs named there), so update them when the measurements change. The "before"
# replay (the S22 one) adds a second p150a row to the categories, to show what has been removed since.
# Args: <p150a replay.json> <a30 tf32 profile.json> <a30 fp32 profile.json> <out.json> [<before replay.json> <csv dir>]
# Run: $VENV_A30/bin/python -I tools/breakdown-sew18.py $DATA/ttnn-replay-s23-ttl-fold.json \
#        $DATA/a30-profile-tf321-s22.json $DATA/a30-profile-tf320-s22.json out.json \
#        $DATA/ttnn-replay-s22-ttl.json csv/
import csv
import json
import os
import sys

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.model import sew_resnet

replay, a30, a30_fp32 = (json.load(open(p)) for p in sys.argv[1:4])
OUT = sys.argv[4]
BEFORE = json.load(open(sys.argv[5])) if len(sys.argv) > 6 else None
CSV_DIR = sys.argv[6] if len(sys.argv) > 6 else None

# measured step times on p150a (ms): xla-sew18.py, WAIT=cpu, batch 8, level 1, BN=eval, NEURON=ttl LIF_LAYOUT=nhwc4
STEPS = {
    # STEM_PAD=32 STEM_FOLD=1 INPUT_LAYOUT=nhwc STEM_SPLIT=2 STEM_LIF4=1 LIF_GRID=11,10 LIF_RES=1 AVGPOOL=matmul, trace
    # (S34, ..-ttl-res-lif2-mmpool-trace-s34-r{1,2,3}.log): the median of the three runs, each value on its own (the
    # synced step time varies by 1 to 2 ms from run to run: the host CPU idles at 800 MHz between the waits, S31)
    "p150a_pipelined": 9.44,
    "p150a_synced": 14.96,  # median of steps 3..19
    "p150a_python_call": 2.11,
    "p150a_submit": 1.38,  # pipelined: the host returns from the calls, per step
    # the S30 model synced with a SCHED_IDLE spinner on the SMT siblings, which keeps the host CPU awake (S31,
    # ..-ttl-res-trace-s31-spin-r{1,2}.log; 14.20 without)
    "p150a_synced_spin": 11.00,
    # S23's best without trace (S23, ..-ttl-fold-notrace-s23.log; not measured for S24)
    "p150a_notrace_pipelined": 23.09,
    "p150a_notrace_synced": 29.65,
    "p150a_notrace_submit": 21.72,
    # before S23: STEM_PAD=32, 4D make_lif, no trace (S21, ..-ttl-nhwc4-4d-s21-waitcpu.log)
    "p150a_before_pipelined": 31.39,
    "p150a_before_synced": 41.10,
}
# the S23 to S34 steps one after another: measured step times (ms) and the replayed device busy time (ms) of each IR
# (None: no replay; the replay runs the LIF on all 110 cores, while the S25 model ran it on 64, D28)
HISTORY = [
    ("始める前（S21・S22）", 41.10, 31.39, 30.36),  # ttnn-replay-s22-ttl.json
    ("layer4 の LIF を [1, T, R, C] で渡す", 34.55, 28.28, 20.34),  # ..-split-s23-waitcpu.log, ttnn-replay-s23-ttl.json
    ("入力をホストで NHWC に並べて渡す", 34.59, 28.32, 16.30),  # ..-split-inhwc-s23-waitcpu.log, ..-s23-ttl-inhwc.json
    ("trace を使う", 22.72, 17.09, 16.30),  # ..-split-inhwc-trace-s23.log (same IR)
    ("stem の conv を 2×2 で折りたたむ", 19.63, 14.51, 13.77),  # ..-fold-trace-s23.log, ttnn-replay-s23-ttl-fold.json
    ("stem の conv を時刻ごとの 4 つに分ける", 18.25, 12.66, 11.21),  # ..-split4-in4-trace-s24.log, ..-s24-split4-in4.json
    ("stem の LIF が 4 つの conv の出力を直接読む", 17.29, 11.87, 11.09),  # ..-lif4-in4-trace-s24*.log, ..-s24-lif4-in4.json
    ("LIF の計算を減らし、ブロックを広げる", 15.21, 10.54, None),  # ..-lif4-in4-trace-s25-r{1,2,3}.log (64 cores)
    ("LIF を 110 コアで動かす", 15.76, 10.18, 9.89),  # ..-grid-trace-s26-r{1,2,3}.log, ttnn-replay-s26-grid.json
    ("LIF のスパイクを 1 回の比較にする", 15.53, 10.10, 9.84),  # ..-grid-trace-s27-r{1,2,3}.log, ttnn-replay-s27.json
    ("残差の足し算を LIF に取り込む", 14.20, 9.59, 9.35),  # ..-res-trace-s30-r{1,2,3}.log, sew18-s30-res.json
    ("stem の conv を 16 枚 × 2 にする", 14.90, 9.48, 9.23),  # ..-res-lif2-trace-s32-r{1,2,3}.log, sew18-s32-lif2.json
    # the replay leaves out one reshape it cannot run (S34)
    ("平均プーリングを行列積にする", 14.96, 9.44, 9.19),  # ..-mmpool-trace-s34-r{1,2,3}.log, sew18-s34-mmpool.json
]
ORDER = ["入力の準備", "conv（stem）", "conv", "BN", "LIF", "並べ替え（layer4 の LIF の前後）", "プーリングと全結合",
         "残差の足し算とその他"]

model = sew_resnet.sew_resnet18(cnf="ADD", num_classes=1000, spiking_neuron=neuron.LIFNode, tau=2.0,
                                detach_reset=True, step_mode="m")
convs = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Conv2d"]
conv_key = {n: (m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0]) for n, m in convs}
neurons = [n for n, m in model.named_modules() if isinstance(m, neuron.BaseNode)]
neuron_c = {}
for n, m in model.named_modules():  # channels a neuron sees = out_channels of the conv before it
    if isinstance(m, neuron.BaseNode):
        stage = n.rsplit(".", 1)[0] if "." in n else ""
        neuron_c[n] = 64 if n in ("sn1",) or n.startswith("layer1") else int(
            {"layer2": 128, "layer3": 256, "layer4": 512}[n.split(".")[0]])

def classify(replay):
    """p150a ops in execution order -> categories and layer names, and the busy time per category (us)."""
    ops = replay["ops"]
    first_pool = next(i for i, o in enumerate(ops) if o["op"] == "ttnn.max_pool2d")
    # the average pooling: a mean, or a matmul with a 0/1 matrix since S34 (AVGPOOL=matmul)
    mean_i = next(i for i, o in enumerate(ops) if o["op"] in ("ttnn.mean", "ttnn.matmul"))
    unused_convs = [n for n, _ in convs]
    p_ops = []
    for i, o in enumerate(ops):
        us = o.get("device_us")
        cat, layer = None, None
        if i < first_pool and o["op"] in ("ttnn.conv2d", "ttnn.concat"):
            # the convs before the maxpool are the stem: padded to 32 channels (D24), maybe folded to 4x4/1
            # (STEM_FOLD, S23) and split into several convs, whose outputs may be concatenated (STEM_SPLIT, S24)
            layer, cat = "conv1", "conv（stem）"
        elif i < first_pool and o["op"] not in ("ttnn.generic", "ttnn.typecast"):
            cat = "入力の準備"  # tiling the input (and slicing it up for a split stem)
        elif o["op"] == "ttnn.conv2d":
            sh = o["desc"].split()
            cin, cout = (int(v) for v in sh[0].split("->"))
            k, s = int(sh[1].split("x")[0]), int(sh[1].split("/")[1])
            layer = next(n for n in unused_convs if conv_key[n] == (cin, cout, k, s))
            unused_convs.remove(layer)
            cat = "conv"
        elif o["op"] == "ttnn.generic":
            cat = "LIF"
        elif o["op"] in ("ttnn.typecast", "ttnn.max_pool2d", "ttnn.to_layout"):
            cat = "プーリングと全結合"  # the maxpool and the type casts around it
        elif o["op"] == "ttnn.add":
            cat = "残差の足し算とその他"
        elif o["op"] in ("ttnn.mean", "ttnn.linear") or i >= mean_i - 1:
            cat = "プーリングと全結合"  # the pooling, the fc, and the relayout right before the pooling
        elif o["op"] in ("ttnn.reshape", "ttnn.permute"):
            cat = "並べ替え（layer4 の LIF の前後）"
        else:
            cat = "残差の足し算とその他"
        p_ops.append({"index": i, "op": o["op"].replace("ttnn.", ""), "desc": o.get("desc", ""), "category": cat,
                      "layer": layer, "device_us": us, "envelope_us": o.get("envelope_us"),
                      "programs": o.get("programs"), "error": o.get("error")})

    # the generics come in execution order, which is not the module order for blocks with a downsample: match LIF
    # layers by channel count, in order. A LIF that adds the residual (S30) is the sn2 of its block, which runs after
    # the downsample_sn
    by_c = {}
    for n in neurons:
        by_c.setdefault(neuron_c[n], []).append(n)
    has_res = any("LIF+add" in r["desc"] for r in p_ops)
    for r in p_ops:
        if r["op"] == "generic":
            shape = r["desc"].split("[")[1].rstrip("]").split(",")
            c = int(shape[-1])
            if c == 128 and int(shape[-2]) == 6272:
                c = 512  # before S23, layer4 ran in [T*R, 128] tiles (N*H*W = 392 is not a multiple of 32)
            res = "LIF+add" in r["desc"]
            r["layer"] = next(n for n in by_c[c] if not has_res or n.endswith(".sn2") == res)
            by_c[c].remove(r["layer"])
    cats = {}
    for r in p_ops:
        cats[r["category"]] = cats.get(r["category"], 0.0) + (r["device_us"] or 0.0)
    return p_ops, cats


p_ops, p_cats = classify(replay)


# Lower bounds of the p150a step (S28). Ops run one after another, and each takes at least
# max(bytes it reads and writes / DRAM bandwidth, FLOPs / peak); the bound of a scenario is the sum over its ops.
# Bytes come from the replayed shapes (fp32 unless the op works in bf16), FLOPs from the conv shapes (2 * MACs).
DRAM_SPEC = 512e9  # GDDR6, 256 bit x 16 Gbps; 511 GB/s measured for reads alone (tt-metal's DRAM read microbenchmark on this card)
DRAM_MIXED = 390e9  # half reads, half writes as measured: a copy-only LIF 330-400 GB/s, a ttnn op 366 GB/s (S25, S27)
PEAK_HIFI4 = 664e12 / 4 * 110 / 120  # spec 664 TFLOPS (BFP8, LoFi, 120 cores); HiFi4 a quarter; 110 cores compute
CONV_FIDELITY_PEAKS = {"LoFi": 664e12 * 110 / 120, "HiFi4": PEAK_HIFI4}


def nel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def op_cost(o):
    """(bytes read and written, FLOPs) of one replayed op."""
    op, d, ins, outs = o["op"], str(o.get("desc", "")), o["in_shapes"], o["out_shapes"]
    if op == "ttnn.conv2d":
        cin, cout = (int(v) for v in d.split()[0].split("->"))
        k = int(d.split()[1].split("x")[0])
        return 4 * (sum(nel(s) for s in ins) + nel(outs[0])), 2.0 * outs[0][2] * cout * cin * k * k
    if op == "ttnn.generic":  # the LIF: its inputs and its out buffer (the last operand)
        return 4 * sum(nel(s) for s in ins), 0.0
    if op == "ttnn.typecast":  # fp32 -> bf16
        return 4 * nel(ins[0]) + 2 * nel(outs[0]), 0.0
    if op == "ttnn.max_pool2d":  # runs in bf16
        return 2 * (nel(ins[0]) + nel(outs[0])), 0.0
    if op == "ttnn.to_layout":
        src, dst = d.split("->") if "->" in d else (d, d)
        return (2 if "bf16" in src else 4) * nel(ins[0]) + (2 if "bf16" in dst else 4) * nel(outs[0]), 0.0
    if op == "ttnn.linear":
        return 4 * (sum(nel(s) for s in ins) + nel(outs[0])), 2.0 * ins[0][0] * ins[0][1] * ins[1][0]
    return 4 * (sum(nel(s) for s in ins) + sum(nel(s) for s in outs)), 0.0


def bounds(replay, ops_cat):
    """scenario -> category -> lower bound (us)."""
    out = {k: {} for k in ("now_mixed", "now_spec", "next_spec", "conv_only")}
    add = lambda sc, cat, us: out[sc].__setitem__(cat, out[sc].get(cat, 0.0) + us)
    for o, r in zip(replay["ops"], ops_cat):
        cat = r["category"]
        b, fl = op_cost(o)
        add("now_mixed", cat, max(b / DRAM_MIXED, fl / PEAK_HIFI4) * 1e6)
        add("now_spec", cat, max(b / DRAM_SPEC, fl / PEAK_HIFI4) * 1e6)
        if fl and o["op"] == "ttnn.conv2d":
            add("conv_only", cat, fl / PEAK_HIFI4 * 1e6)
        # without the moves and casts: the input goes on the device tiled, the layer4 LIF takes its rows as they
        # are, the stem LIF writes bf16 spikes (no typecast before the maxpool), and the LIF reads the residual and
        # writes the sum (one more read for the LIF, no add; done since S30). S33: tt-xla 1.4.0 cannot do the first
        # two
        if cat in ("入力の準備", "並べ替え（layer4 の LIF の前後）") or o["op"] == "ttnn.typecast":
            continue
        if o["op"] == "ttnn.add":
            b, cat = b / 3, "LIF"
        elif o["op"] == "ttnn.generic" and r["layer"] == "sn1":  # the stem LIF: its out buffer is the last operand
            b -= 2 * nel(o["in_shapes"][-1])
        add("next_spec", cat, max(b / DRAM_SPEC, fl / PEAK_HIFI4) * 1e6)
    conv_gflop = sum(op_cost(o)[1] for o in replay["ops"] if o["op"] == "ttnn.conv2d") / 1e9
    return out, conv_gflop


p_bounds, conv_gflop = bounds(replay, p_ops)
lif_bytes = sum(op_cost(o)[0] for o in replay["ops"] if o["op"] == "ttnn.generic")
BOUND_ROWS = [  # scenario key, label in the chart
    ("now_mixed", "下限: 今の並び・実測の帯域"),
    ("now_spec", "下限: 今の並び・仕様の帯域"),
    ("next_spec", "下限: 並べ替えと型の変換を除く"),
    ("conv_only", "下限: conv の演算だけ"),
]


def a30_cats(prof):
    cats = {}
    names = {"Conv2d": "conv", "LIFNode": "LIF", "BatchNorm2d": "BN", "MaxPool2d": "プーリングと全結合",
             "AdaptiveAvgPool2d": "プーリングと全結合", "Linear": "プーリングと全結合"}
    for r in prof["layers"]:
        cat = "conv（stem）" if r["layer"] == "conv1" else names[r["type"]]
        cats[cat] = cats.get(cat, 0.0) + r["gpu_us"]
    cats["残差の足し算とその他"] = prof["gpu_kernels_total_us"] - prof["gpu_in_layers_us"]
    return cats


layout_kernels = sum(us for k, us in a30["top_kernels_us"] if "nchwToNhwc" in k or "nhwcToNchw" in k)

conv_rows = []
a30_layers = {r["layer"]: r["gpu_us"] for r in a30["layers"]}
a30f_layers = {r["layer"]: r["gpu_us"] for r in a30_fp32["layers"]}
for r in p_ops:
    if r["op"] == "conv2d":
        if conv_rows and r["layer"] == "conv1" == conv_rows[-1]["layer"]:  # a split stem: one row for all its convs
            row = conv_rows[-1]
            row["n"] += 1
            for key, rkey in (("p150a_us", "device_us"), ("p150a_envelope_us", "envelope_us"), ("programs", "programs")):
                row[key] += r[rkey]
            row["shape"] = row["shape"].split("、")[0] + f"、p150a では {row['on_dev']} を {row['n']} 回"
            continue
        cin, cout, k, s = conv_key[r["layer"]]
        shape = f"{cin}→{cout} {k}×{k}/{s}"
        sh = r["desc"].split()
        on_dev = f"{sh[0].replace('->', '→')} {sh[1].replace('x', '×', 1)}"
        if on_dev != shape:  # the stem: padded (and folded) on p150a
            shape += f"、p150a では {on_dev}"
        conv_rows.append({"layer": r["layer"], "shape": shape, "p150a_us": r["device_us"],
                          "p150a_envelope_us": r["envelope_us"], "programs": r["programs"],
                          "a30_tf32_us": a30_layers.get(r["layer"]), "a30_fp32_us": a30f_layers.get(r["layer"]),
                          "n": 1, "on_dev": on_dev})
lif_rows = [{"layer": r["layer"], "p150a_us": r["device_us"], "a30_us": a30_layers.get(r["layer"])}
            for r in p_ops if r["op"] == "generic"]

out = {
    "steps_ms": {
        "a30_tf32_synced": a30["synced_step_ms"], "a30_tf32_host_call": a30["host_call_ms"],
        "a30_fp32_synced": a30_fp32["synced_step_ms"], **STEPS,
    },
    "a30_tf32_cats_us": a30_cats(a30), "a30_fp32_cats_us": a30_cats(a30_fp32),
    "a30_tf32_kernels_us": a30["gpu_kernels_total_us"], "a30_fp32_kernels_us": a30_fp32["gpu_kernels_total_us"],
    "a30_tf32_layout_kernels_us": layout_kernels,
    "p150a_cats_us": p_cats, "p150a_busy_us": sum(p_cats.values()),
    "p150a_before_cats_us": classify(BEFORE)[1] if BEFORE else None, "history": HISTORY,
    "p150a_ops": p_ops, "conv_rows": conv_rows, "lif_rows": lif_rows,
    "p150a_bounds_us": p_bounds, "conv_gflop": conv_gflop,
    "a30_top_kernels_us": a30["top_kernels_us"][:12],
}
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k.endswith("cats_us") or k == "steps_ms" or k.endswith("busy_us")},
                 ensure_ascii=False, indent=1))

if CSV_DIR:
    os.makedirs(CSV_DIR, exist_ok=True)
    total = lambda d, k: sum(o.get(k, 0.0) for o in d["ops"] if k in o) / 1e3
    st = out["steps_ms"]
    steps = [
        ("a30_tf32", "A30（TF32、cudnn.benchmark）1 ステップ", round(st["a30_tf32_synced"], 2), "a30-sew18-profile.py、出力を読み戻すまで、30 回の中央値"),
        ("a30_tf32_host", "A30 の Python の呼び出し（GPU を待たない）", round(st["a30_tf32_host_call"], 2), "a30-sew18-profile.py"),
        ("a30_tf32_kernels", "A30 の GPU カーネルの合計", round(a30["gpu_kernels_total_us"] / 1e3, 2), "torch.profiler、10 ステップの平均"),
        ("a30_fp32", "A30（厳密な fp32、ベンチマークの条件）1 ステップ", round(st["a30_fp32_synced"], 2), "a30-sew18-profile.py"),
        ("p150a_pipelined", "p150a まとめて同期（trace、S34 の版）", st["p150a_pipelined"], "xla-sew18.py、19 ステップを続けて流して最後に同期、3 回の中央値（S34）"),
        ("p150a_synced", "p150a 1 ステップごとに同期（同じ版）", st["p150a_synced"], "xla-sew18.py WAIT=cpu、3〜19 ステップ目の中央値、3 回の中央値（S34）。ホストの CPU は既定の省電力のまま"),
        ("p150a_synced_spin", "p150a 1 ステップごとに同期（S30 の版、ホストの CPU を起こしておいた場合）", st["p150a_synced_spin"], "SMT の兄弟の CPU に低優先度の空回りを置いて測った、2 回（S31）。同じ版で空回りなしは 14.2 ms"),
        ("p150a_submit", "p150a まとめて同期のときのホストの送り出し（1 ステップあたり）", st["p150a_submit"], "xla-sew18.py の pipelined_split（S34）"),
        ("p150a_python", "p150a の Python の呼び出し", st["p150a_python_call"], "xla-sew18.py の call（S34）"),
        ("p150a_notrace_pipelined", "p150a trace なし（S23 の版）: まとめて同期", st["p150a_notrace_pipelined"], "xla-sew18.py（S23）"),
        ("p150a_notrace_synced", "p150a trace なし（S23 の版）: 1 ステップごとに同期", st["p150a_notrace_synced"], "xla-sew18.py（S23）"),
        ("p150a_notrace_submit", "p150a trace なし（S23 の版）: ホストの送り出し（1 ステップあたり）", st["p150a_notrace_submit"], "xla-sew18.py の pipelined_split（S23）"),
        ("p150a_busy", "p150a デバイスの稼働時間の合計（流し直し）", round(total(replay, "device_us"), 2), "ttnn-replay-ir.py、コンテナの ttnn とプロファイラ（流せなかった演算は入らない）"),
        ("p150a_envelope", "p150a 演算ごとの区間の合計（流し直し）", round(total(replay, "envelope_us"), 2), "ttnn-replay-ir.py"),
        ("p150a_before_pipelined", "始める前（S21）: まとめて同期", st["p150a_before_pipelined"], "xla-sew18.py（S21）"),
        ("p150a_before_synced", "始める前（S21）: 1 ステップごとに同期", st["p150a_before_synced"], "xla-sew18.py（S21）"),
        ("p150a_before_busy", "始める前（S22）: デバイスの稼働時間の合計（流し直し）", round(total(BEFORE, "device_us"), 2), "ttnn-replay-ir.py"),
    ]
    with open(os.path.join(CSV_DIR, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "ms", "how"])
        w.writerows(steps)
    with open(os.path.join(CSV_DIR, "categories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "category", "order", "ms"])
        for plat, cats in (("A30 TF32", out["a30_tf32_cats_us"]), ("A30 fp32", out["a30_fp32_cats_us"]),
                           ("p150a", out["p150a_cats_us"]), ("p150a before", out["p150a_before_cats_us"])):
            for i, c in enumerate(ORDER):
                w.writerow([plat, c, i, round(cats.get(c, 0.0) / 1e3, 3)])
    with open(os.path.join(CSV_DIR, "ops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "op", "detail", "category", "layer", "busy_us", "envelope_us", "programs"])
        for o in p_ops:
            w.writerow([o["index"], o["op"], o["desc"] if not o.get("error") else "再生できず（タイルの reshape）", o["category"],
                        o["layer"] or "", "" if o["device_us"] is None else round(o["device_us"], 1),
                        "" if o["envelope_us"] is None else round(o["envelope_us"], 1), o["programs"] or ""])
    with open(os.path.join(CSV_DIR, "conv_layers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "layer", "shape", "p150a_busy_us", "p150a_envelope_us", "programs", "a30_tf32_us", "a30_fp32_us"])
        for i, r in enumerate(conv_rows):
            w.writerow([i, r["layer"], r["shape"], round(r["p150a_us"], 1), round(r["p150a_envelope_us"], 1), r["programs"],
                        round(r["a30_tf32_us"], 1), round(r["a30_fp32_us"], 1)])
    with open(os.path.join(CSV_DIR, "lif_layers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "layer", "p150a_us", "a30_us"])
        for i, r in enumerate(lif_rows):
            w.writerow([i, r["layer"], round(r["p150a_us"], 1), round(r["a30_us"], 1)])
    with open(os.path.join(CSV_DIR, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "stage", "synced_ms", "pipelined_ms", "busy_ms"])
        for i, (stage, synced, pipelined, busy) in enumerate(HISTORY):
            w.writerow([i, stage, synced, pipelined, busy])
    with open(os.path.join(CSV_DIR, "bounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "scenario", "label", "category", "cat_order", "ms"])
        for i, (key, label) in enumerate(BOUND_ROWS):
            for j, c in enumerate(ORDER):
                if c in p_bounds[key]:
                    w.writerow([i, key, label, c, j, round(p_bounds[key][c] / 1e3, 3)])
    with open(os.path.join(CSV_DIR, "bound_params.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "value", "unit", "how"])
        w.writerows([
            ("dram_spec", "DRAM の帯域（仕様）", round(DRAM_SPEC / 1e9), "GB/s", "GDDR6 256 bit × 16 Gbps。読み出しだけの実測は 511 GB/s"),
            ("dram_mixed", "DRAM の帯域（読み書き半々の実測）", round(DRAM_MIXED / 1e9), "GB/s", "LIF の転送だけの版 330〜400 GB/s、ttnn の演算 366 GB/s"),
            ("peak_hifi4", "演算性能（HiFi4、110 コア）", round(PEAK_HIFI4 / 1e12), "TFLOPS", "仕様 664 TFLOPS（BFP8・LoFi、120 コア）の 4 分の 1 を 110 コア分。BF16・HiFi4 の matmul の実測は 130 TFLOPS"),
            ("peak_lofi", "演算性能（LoFi、110 コア）", round(CONV_FIDELITY_PEAKS["LoFi"] / 1e12), "TFLOPS", "仕様 664 TFLOPS の 110 コア分"),
            ("conv_gflop", "conv の演算量（1 ステップ）", round(conv_gflop, 1), "GFLOP", "23 個の conv の 2 × 積和の数（stem は 32 チャネルに埋めて折りたたんだ形）"),
            ("conv_lofi_ms", "conv の演算だけの下限（LoFi）", round(conv_gflop / (CONV_FIDELITY_PEAKS["LoFi"] / 1e12), 3), "ms", "精度を落とした場合"),
            ("lif_mb", "LIF の読み書きの量（1 ステップ）", round(lif_bytes / 1e6), "MB", "20 層の入力、残差、出力（fp32）。流し直した演算の形から"),
            ("lif_mixed_ms", "LIF の下限（読み書き半々の実測の帯域）", round(lif_bytes / DRAM_MIXED * 1e3, 2), "ms", "LIF の読み書きの量 ÷ 読み書き半々の帯域"),
        ])
