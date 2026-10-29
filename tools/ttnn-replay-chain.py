# Replay a TTNN IR's traced function (as exported by tt-xla with trace) end to end with ttnn in the tt-lang container,
# each op on the outputs of the ops before it, and time each op on the device with tt-metal's device profiler (S29).
# Unlike ttnn-replay-ir.py (each op on fresh DRAM inputs), the tensors flow as in the IR, the IR's deallocates are
# kept, and where the outputs live is a choice, so what fits in L1 and what it saves can be seen for the whole step.
# The LIF outputs, which tt-xla passes in as preallocated arguments alive for the whole step, are allocated at their
# ttnn.generic and freed after their last use, as a hand-written model would. Inputs and the fc weights are random,
# in DRAM; conv weights are random and prepared on the first run. ttnn.generic is the tt-lang LIF (make_lif, or
# make_lif4 with five operands).
# Policy (comma-separated key=value):
#   mem=dram    where op outputs go: dram (as the IR); l1 (L1 interleaved); l1s (a conv's output stays sharded where
#               the conv puts it, and the LIF and add after it write with the same sharding; other ops L1 interleaved).
#               An op that fails there falls back to L1 interleaved, then DRAM, then DRAM after moving every live L1
#               tensor to DRAM (the move is timed with the op); decided on the first run and kept
#   l1_from=0   ops before this index (the stem: its four conv outputs and the LIF4 output, 103 MB, do not fit) in DRAM
#   dbuf=0      1: enable_act_double_buffer on every conv (falls back to off where it fails)
#   dram_ops=   op indices forced to DRAM, joined by ":"
#   lif_tiles=32  the block size limit of make_lif (smaller blocks, smaller CBs in L1; make_lif4 keeps 32)
# On the first run every LIF output is compared with the fp32 reference on the CPU (spikes, exact).
# Args: <ttnn.mlir> <out.json> <policy> [reps] [T]
# Run (inside the tt-lang container, with TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal; run from a path mounted in the container):
#   docker exec -i -e TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal <tt-lang container> bash -c "cd /tmp && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
#     python3 $PWD/tools/ttnn-replay-chain.py $IRDIR/sew18.mlir $IRDIR/out.json mem=l1"
import json
import os
import re
import statistics
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttl_lif import lif_reference, make_lif, make_lif4  # noqa: E402

IR, OUT = sys.argv[1], sys.argv[2]
POLICY = dict(kv.split("=", 1) for kv in sys.argv[3].split(",") if kv)
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
T = int(sys.argv[5]) if len(sys.argv) > 5 else 4
MEM = POLICY.get("mem", "dram")
DBUF = POLICY.get("dbuf") == "1"
DRAM_OPS = {int(i) for i in POLICY.get("dram_ops", "").split(":") if i}
L1_FROM = int(POLICY.get("l1_from", "0"))
LIF_TILES = int(POLICY.get("lif_tiles", "32"))

text = open(os.path.expanduser(IR)).read()
tile_layouts = {m.group(1): "!ttcore.tile" in m.group(2) for m in re.finditer(r"^#(ttnn_layout\d*) = (.*)$", text, re.M)}
start = text.index("func.func private @trace_0_main(")
end = text.find("func.func", start + 10)
body = text[start:end]
TYPE = r"tensor<((?:\d+x)+)(f32|bf16|si32|ui32|u32|i32)(?:, #(ttnn_layout\d*))?>"
DT = {"f32": ttnn.float32, "bf16": ttnn.bfloat16}


def typ(m):
    return [int(d) for d in m.group(1)[:-1].split("x")], m.group(2), tile_layouts.get(m.group(3), True)


def types(s):
    return [typ(m) for m in re.finditer(TYPE, s)]


def ints(attrs, key):
    m = re.search(key + r" = array<i(?:32|64): ([0-9, ]+)>", attrs)
    return [int(v) for v in m.group(1).split(",")] if m else None


def ints_list(attrs, key):
    m = re.search(key + r" = \[([-0-9 :i,]+)\]", attrs)
    return [int(v) for v in m.group(1).replace(" : i32", "").split(",")] if m else None


def one(attrs, key):
    m = re.search(key + r" = (-?[0-9]+|true|false)", attrs)
    return None if not m else (m.group(1) == "true" if m.group(1) in ("true", "false") else int(m.group(1)))


arg_types = {m.group(1): types(m.group(2))[0]
             for m in re.finditer(r"(%arg\d+): (" + TYPE + ")", body[:body.index("{\n")])}
ops = []  # in order; deallocates are kept as ops of their own
for line in body.splitlines():
    m = re.search(r'^\s*(?:(%\w+) = )?"(ttnn\.[a-z_0-9]+)"\(([^)]*)\)', line)
    if not m or m.group(2) == "ttnn.get_device":
        continue
    name = m.group(2)
    operands = [v.strip() for v in m.group(3).split(",") if v.strip() and v.strip() != "%0"]
    attrs = "" if name == "ttnn.generic" else (line[line.find("<{"):line.find("}>") + 2] if "<{" in line else "")
    sig = line[line.rfind(" : (") + 4:] if " : (" in line else ""
    ins_s, _, out_s = sig.partition(") -> ")
    o = {"op": name, "attrs": attrs, "args": operands, "ins": types(ins_s), "outs": types(out_s)}
    if name == "ttnn.generic":  # writes into its last operand
        o["result"], o["args"] = operands[-1], operands[:-1]
    else:
        o["result"] = m.group(1)
    ops.append(o)
ret = re.search(r"return (%\w+)", body).group(1)
last_use = {}
for i, o in enumerate(ops):
    for v in o["args"]:
        last_use[v] = i
lif_outs = {o["result"] for o in ops if o["op"] == "ttnn.generic"}

device = ttnn.open_device(device_id=0, l1_small_size=32768)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
csv_pos = [0, None]  # bytes already read, chip frequency


def new_spans():
    """(start, end) per program (run host ID) over *-KERNEL zones in the CSV lines added since the last call."""
    ttnn.ReadDeviceProfiler(device)
    st, en = {}, {}
    if os.path.getsize(CSV) < csv_pos[0]:
        csv_pos[0] = 0
    with open(CSV) as fh:
        if csv_pos[0] == 0:
            csv_pos[1] = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
            fh.readline()
        else:
            fh.seek(csv_pos[0])
        for line in fh:
            p = line.split(",", 12)
            if len(p) < 12 or not p[10].strip().endswith("-KERNEL"):
                continue
            rid, t = int(p[7]), int(p[5])
            if p[11].strip() == "ZONE_START":
                st[rid] = min(st.get(rid, t), t)
            else:
                en[rid] = max(en.get(rid, t), t)
        csv_pos[0] = fh.tell()
    return [(st[r], en[r]) for r in sorted(st) if r in en], csv_pos[1]


def host_tensor(shape, dt, tile, scale=1.0):
    return ttnn.from_torch(torch.randn(shape) * scale, dtype=DT[dt],
                           layout=ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)


consts = {}  # function arguments read as data: the inputs and the fc weights (conv weights are prepared per conv)
for o in ops:
    if o["op"] in ("ttnn.to_layout", "ttnn.linear"):
        for v in o["args"]:
            if v.startswith("%arg") and v not in consts:
                consts[v] = host_tensor(*arg_types[v], scale=0.05 if o["op"] == "ttnn.linear" else 1.0)
prepared = {}  # op index -> (conv kwargs, prepared weights, bias) once the first run has settled them
choice = {}  # op index -> where its output went on the first run


def run_op(i, o, env, where):
    """run op i on the tensors in env with its output placed by where (S, L1, DRAM, MOVE); returns the output."""
    name, a, ins, outs = o["op"], o["attrs"], o["ins"], o["outs"]
    if where == "MOVE":  # make room: every live L1 tensor to DRAM, then as DRAM
        for v, t in list(env.items()):
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                env[v] = ttnn.to_memory_config(t, DRAM)
                ttnn.deallocate(t, force=False)
    x = [env[v] if v in env else consts.get(v) for v in o["args"]]
    mc = L1 if where in ("S", "L1") else DRAM
    if where == "S" and name == "ttnn.conv2d":
        mc = None  # where the conv puts it
    elif where == "S" and name in ("ttnn.generic", "ttnn.add") and x[0].is_sharded():
        mc = x[0].memory_config()
    if name == "ttnn.conv2d":
        if i in prepared:
            kw, wd, bd = prepared[i]
            return ttnn.conv2d(input_tensor=x[0], weight_tensor=wd, bias_tensor=bd, **dict(kw, memory_config=mc))
        cin, cout = one(a, "in_channels"), one(a, "out_channels")
        kh, kw_ = ints(a, "kernel_size")
        sl = re.search(r"conv2d_slice_config<([a-z0-9_]+)", a).group(1)
        nsl = int(re.search(r"conv2d_slice_config<[a-z0-9_]+, (\d+)", a).group(1))
        slice_config = (ttnn.Conv2dL1FullSliceConfig if sl == "l1_full" else ttnn.Conv2dSliceConfig(
            slice_type={"dram_height": ttnn.Conv2dDRAMSliceHeight, "dram_width": ttnn.Conv2dDRAMSliceWidth}[sl],
            num_slices=nsl))
        last = None
        for dbuf in ([True, False] if DBUF else [False]):
            cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.float32, act_block_h_override=one(a, "act_block_h_override") or 0,
                                    enable_kernel_stride_folding=False, config_tensors_in_dram=True,
                                    deallocate_activation=False, enable_act_double_buffer=dbuf)
            kw = dict(in_channels=cin, out_channels=cout, device=device, batch_size=one(a, "batch_size"),
                      input_height=one(a, "input_height"), input_width=one(a, "input_width"), kernel_size=(kh, kw_),
                      stride=tuple(ints(a, "stride")), padding=tuple(ints(a, "padding")),
                      dilation=tuple(ints(a, "dilation")), groups=one(a, "groups"), conv_config=cfg,
                      slice_config=slice_config)
            wt = ttnn.from_torch(torch.randn(cout, cin, kh, kw_) / (cin * kh * kw_) ** 0.5, dtype=ttnn.float32)
            bs = ttnn.from_torch(torch.randn(1, 1, 1, cout) * 0.1, dtype=ttnn.float32)
            try:
                y, (wd, bd) = ttnn.conv2d(input_tensor=x[0], weight_tensor=wt, bias_tensor=bs,
                                          return_weights_and_bias=True, memory_config=mc, **kw)
                if i in prepared:  # an earlier placement's weights
                    ttnn.deallocate(prepared[i][1])
                    ttnn.deallocate(prepared[i][2])
                prepared[i] = (kw, wd, bd)
                o["dbuf"] = dbuf
                return y
            except Exception as e:  # try the next setting; the caller falls back to DRAM if none works
                last = e
        raise last
    if name == "ttnn.generic":
        shape, dt, tile = outs[0] if outs else ins[-1]
        out = ttnn.empty(shape, DT[dt], ttnn.TILE_LAYOUT, device, mc)
        if "kernel" not in o:
            o["kernel"] = make_lif4() if len(x) == 4 else make_lif(T, max_tiles=LIF_TILES)  # the stem's stays in DRAM
        o["kernel"](*x, out)
        if "mismatch" not in o:  # first run: the spikes against the CPU
            xin = torch.cat([ttnn.to_torch(t).float().reshape(-1, shape[-1]) for t in x]) if len(x) == 4 else \
                ttnn.to_torch(x[0]).float().reshape(-1, shape[-1])
            ref, _ = lif_reference(xin, T)
            o["mismatch"] = int((ttnn.to_torch(out).float().reshape(-1, shape[-1]) != ref).sum())
            o["out_mem"] = str(out.memory_config().memory_layout).split(".")[-1]
        return out
    if name == "ttnn.to_layout":
        shape, dt, tile = outs[0]
        return ttnn.to_layout(x[0], ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT, dtype=DT[dt], memory_config=mc)
    if name == "ttnn.typecast":
        return ttnn.typecast(x[0], DT[outs[0][1]], memory_config=mc)
    if name == "ttnn.add":
        return ttnn.add(x[0], x[1], memory_config=mc)
    if name == "ttnn.reshape":
        return ttnn.reshape(x[0], ints_list(a, "shape"), memory_config=mc)
    if name == "ttnn.max_pool2d":
        return ttnn.max_pool2d(x[0], batch_size=one(a, "batch_size"), input_h=one(a, "input_height"),
                               input_w=one(a, "input_width"), channels=one(a, "channels"),
                               kernel_size=ints(a, "kernel_size"), stride=ints(a, "stride"), padding=ints(a, "padding"),
                               dilation=ints(a, "dilation"), ceil_mode=bool(one(a, "ceil_mode")), memory_config=mc,
                               reallocate_halo_output=False)
    if name == "ttnn.mean":
        return ttnn.mean(x[0], dim=ints_list(a, "dim_arg"), keepdim=bool(one(a, "keep_dim")), memory_config=mc)
    if name == "ttnn.linear":
        return ttnn.linear(x[0], x[1], bias=x[2] if len(x) > 2 else None, transpose_b=True, memory_config=mc)
    raise NotImplementedError(name)


def step(timed):
    """one pass over the ops; per op: device busy time (us), programs, where the output went."""
    env, rows = {}, []
    for i, o in enumerate(ops):
        if o["op"] == "ttnn.deallocate":
            v = o["args"][0]
            if v in env:
                ttnn.deallocate(env.pop(v), force=False)
            continue
        if i not in choice:  # first run: try the policy's place, then fall back
            tries = (["DRAM"] if MEM == "dram" or i in DRAM_OPS or i < L1_FROM else
                     ["S", "L1", "DRAM", "MOVE"] if MEM == "l1s" else ["L1", "DRAM", "MOVE"])
            for where in tries:
                try:
                    env[o["result"]] = run_op(i, o, env, where)
                    choice[i] = where
                    break
                except Exception as e:
                    o.setdefault("errors", []).append(f"{where}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            else:
                raise RuntimeError(f"op {i} {o['op']} failed: {o['errors']}")
        else:
            env[o["result"]] = run_op(i, o, env, choice[i])
        ttnn.synchronize_device(device)
        sp, freq = new_spans()
        rows.append((i, sum(e - s for s, e in sp) / freq, len(sp)))
        for v in o["args"]:  # LIF outputs are freed after their last use (the IR keeps them as arguments)
            if v in lif_outs and last_use.get(v) == i and v in env:
                ttnn.deallocate(env.pop(v), force=False)
    for v in list(env):
        ttnn.deallocate(env.pop(v), force=False)
    return rows


try:
    step(False)  # compile, decide where each output goes, prepare the conv weights
    new_spans()
    reps = [step(True) for _ in range(REPS)]
finally:
    ttnn.close_device(device)

CAT = {"ttnn.conv2d": "conv", "ttnn.generic": "LIF", "ttnn.add": "add", "ttnn.to_layout": "to_layout",
       "ttnn.reshape": "reshape", "ttnn.typecast": "typecast", "ttnn.max_pool2d": "maxpool", "ttnn.mean": "mean",
       "ttnn.linear": "linear"}
results, by_cat = [], {}
for k, (i, _, _) in enumerate(reps[0]):
    o = ops[i]
    us = statistics.median(r[k][1] for r in reps)
    r = {"index": i, "op": o["op"], "where": choice[i], "device_us": us, "programs": reps[0][k][2],
         "in_shapes": [t[0] for t in o["ins"]], "dbuf": o.get("dbuf"), "errors": o.get("errors"),
         "lif_mismatch": o.get("mismatch"), "lif_out_mem": o.get("out_mem")}
    results.append(r)
    by_cat[CAT[o["op"]]] = by_cat.get(CAT[o["op"]], 0.0) + us
    print(f"{i:3d} {o['op']:<16} {str(r['in_shapes'][0]):<22} {choice[i]:<4} {us:8.1f} us "
          f"({r['programs']} programs){' dbuf' if o.get('dbuf') else ''}"
          f"{' mismatch %d %s' % (o['mismatch'], o['out_mem']) if 'mismatch' in o else ''}"
          f"{'  [' + '; '.join(o['errors']) + ']' if o.get('errors') else ''}", flush=True)
totals = [sum(r[k][1] for k in range(len(r))) for r in reps]
print(f"policy {sys.argv[3]}: device busy per step {statistics.median(totals) / 1e3:.3f} ms (reps {[round(t / 1e3, 3) for t in totals]}); "
      + ", ".join(f"{c} {v / 1e3:.3f}" for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])), flush=True)
json.dump({"ir": IR, "policy": sys.argv[3], "reps": REPS, "T": T, "step_ms": [t / 1e3 for t in totals],
           "by_category_ms": {c: v / 1e3 for c, v in by_cat.items()}, "ops": results},
          open(os.path.expanduser(OUT), "w"), indent=1)
