# Replay the ops of a TTNN IR's main function (as exported by tt-xla's export_path) one at a time with ttnn in the
# tt-lang container, and time each on the device with tt-metal's device profiler. tt-xla 1.4.0's own tt-metal is built
# without Tracy, so this is how the per-op device time of a tt-xla program is estimated (the container's tt-metal is
# a different version; the conv configs are taken from the IR as they are).
# Each op runs on fresh random inputs of the operand types (shape, dtype, tile or row-major, DRAM interleaved); ops
# are not chained. A conv first runs once with host weights to get them prepared (tt-xla prepares them once, in a
# const-eval function), then is timed with the prepared ones. ttnn.generic is taken to be the tt-lang LIF (make_lif)
# on [1, 1, T*R, C], or make_lif4 when it has five operands, or with three make_lif_add (S30) when the first and
# the last have the same shape and make_lif2 (S32) when not. Device time of one call = sum of the KERNEL spans of the
# programs it ran (busy time); the envelope (first start .. last end) is kept too. An IR exported with trace has its
# ops in @trace_0_main, which is replayed instead of @main.
# Args: <ttnn.mlir> <out.json> [reps] [T]
# Run (inside the tt-lang container, with TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal; run from a path mounted in the container):
#   docker exec -i -e TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal <tt-lang container> bash -c "cd /tmp && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
#     python3 $PWD/tools/ttnn-replay-ir.py $IRDIR/sew18.mlir $IRDIR/sew18.json 5"
import json
import os
import re
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttl_lif import make_lif, make_lif2, make_lif4, make_lif_add  # noqa: E402

IR, OUT = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
T = int(sys.argv[4]) if len(sys.argv) > 4 else 4

text = open(os.path.expanduser(IR)).read()
tile_layouts = {m.group(1): "!ttcore.tile" in m.group(2) for m in re.finditer(r"^#(ttnn_layout\d*) = (.*)$", text, re.M)}
start = text.index("func.func @main(")
if "ttnn.capture_or_execute_trace" in text[start:text.find("func.func", start + 10)]:
    start = text.index("func.func private @trace_0_main(")  # exported with trace: the ops are in the traced function
end = text.find("func.func", start + 10)
body = text[start:end if end > 0 else None]

TYPE = re.compile(r"tensor<((?:\d+x)+)(f32|bf16|si32|ui32|u32|i32)(?:, #(ttnn_layout\d*))?>")
DT = {"f32": ttnn.float32, "bf16": ttnn.bfloat16}


def types(s):
    return [([int(d) for d in m.group(1)[:-1].split("x")], m.group(2), tile_layouts.get(m.group(3), True))
            for m in TYPE.finditer(s)]


def ints(attrs, key):
    m = re.search(key + r" = array<i(?:32|64): ([0-9, ]+)>", attrs)
    return [int(v) for v in m.group(1).split(",")] if m else None


def ints_list(attrs, key):
    m = re.search(key + r" = \[([-0-9 :i,]+)\]", attrs)
    return [int(v) for v in m.group(1).replace(" : i32", "").split(",")] if m else None


def one(attrs, key):
    m = re.search(key + r" = (-?[0-9]+|true|false)", attrs)
    return None if not m else (m.group(1) == "true" if m.group(1) in ("true", "false") else int(m.group(1)))


ops = []
for line in body.splitlines():
    m = re.search(r'"(ttnn\.[a-z_0-9]+)"\(', line)
    if not m or m.group(1) in ("ttnn.deallocate", "ttnn.get_device"):
        continue
    name = m.group(1)
    attrs = line[line.find("<{"):line.find("}>") + 2] if "<{" in line else ""
    if name == "ttnn.generic":
        attrs = ""  # the kernel source is in the attribute; not needed
    sig = line[line.rfind(" : (") + 4:] if " : (" in line else ""
    ins_s, _, out_s = sig.partition(") -> ")
    ops.append({"op": name, "attrs": attrs, "ins": types(ins_s), "outs": types(out_s),
                "loc": (re.search(r"loc\((#loc\d+)\)", line) or [None, None])[1]})

device = ttnn.open_device(device_id=0, l1_small_size=32768)
DRAM = ttnn.DRAM_MEMORY_CONFIG
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


def tensor(shape, dt, tile, scale=1.0, shift=0.0):
    t = torch.randn(shape) * scale + shift
    return ttnn.from_torch(t, dtype=DT[dt], layout=ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT,
                           device=device, memory_config=DRAM)


def build(o):
    """a callable that runs the op once on fresh inputs, and a short description."""
    name, a, ins, outs = o["op"], o["attrs"], o["ins"], o["outs"]
    if name == "ttnn.conv2d":
        x = tensor(*ins[0])
        cin, cout = one(a, "in_channels"), one(a, "out_channels")
        kh, kw = ints(a, "kernel_size")
        wt = ttnn.from_torch(torch.randn(cout, cin, kh, kw) * 0.05, dtype=ttnn.float32)
        bs = ttnn.from_torch(torch.randn(1, 1, 1, cout) * 0.05, dtype=ttnn.float32)
        sl = re.search(r"conv2d_slice_config<([a-z0-9_]+)", a).group(1)
        nsl = int(re.search(r"conv2d_slice_config<[a-z0-9_]+, (\d+)", a).group(1))
        slice_config = (ttnn.Conv2dL1FullSliceConfig if sl == "l1_full" else ttnn.Conv2dSliceConfig(
            slice_type={"dram_height": ttnn.Conv2dDRAMSliceHeight, "dram_width": ttnn.Conv2dDRAMSliceWidth}[sl],
            num_slices=nsl))
        cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.float32, act_block_h_override=one(a, "act_block_h_override") or 0,
                                enable_kernel_stride_folding=False, config_tensors_in_dram=True,
                                deallocate_activation=False)
        kw_args = dict(in_channels=cin, out_channels=cout, device=device, batch_size=one(a, "batch_size"),
                       input_height=one(a, "input_height"), input_width=one(a, "input_width"),
                       kernel_size=(kh, kw), stride=tuple(ints(a, "stride")), padding=tuple(ints(a, "padding")),
                       dilation=tuple(ints(a, "dilation")), groups=one(a, "groups"), conv_config=cfg,
                       slice_config=slice_config, memory_config=DRAM)
        y, (wd, bd) = ttnn.conv2d(input_tensor=x, weight_tensor=wt, bias_tensor=bs, return_weights_and_bias=True,
                                  **kw_args)
        ttnn.deallocate(y)
        desc = f"{cin}->{cout} {kh}x{kw}/{ints(a, 'stride')[0]} {one(a, 'input_height')}x{one(a, 'input_width')} {sl}"
        return (lambda: ttnn.conv2d(input_tensor=x, weight_tensor=wd, bias_tensor=bd, **kw_args)), desc
    if name == "ttnn.generic" and len(ins) == 5:  # make_lif4: four time steps in, one tensor out
        xs = [tensor(*t, scale=1.5, shift=0.5) for t in ins[:4]]
        out = tensor(*ins[4])
        kernel = make_lif4()
        return (lambda: (kernel(*xs, out), None)[1]), f"tt-lang LIF4 4 x {ins[0][0]}"
    if name == "ttnn.generic" and len(ins) == 3 and ins[0][0] != ins[2][0]:  # make_lif2: two time steps per input
        xs = [tensor(*t, scale=1.5, shift=0.5) for t in ins[:2]]
        out = tensor(*ins[2])
        kernel = make_lif2()
        return (lambda: (kernel(*xs, out), None)[1]), f"tt-lang LIF2 2 x {ins[0][0]}"
    if name == "ttnn.generic" and len(ins) == 3:  # make_lif_add: x, the residual, out = s + r
        x = tensor(*ins[0], scale=1.5, shift=0.5)
        r, out = tensor(*ins[1]), tensor(*ins[2])
        kernel = make_lif_add(T)
        return (lambda: (kernel(x, r, out), None)[1]), f"tt-lang LIF+add {ins[0][0]}"
    if name == "ttnn.generic":
        x = tensor(*ins[0], scale=1.5, shift=0.5)
        out = tensor(*ins[1])
        kernel = make_lif(T)
        return (lambda: (kernel(x, out), None)[1]), f"tt-lang LIF {ins[0][0]}"
    if name == "ttnn.add":
        a_, b_ = tensor(*ins[0]), tensor(*ins[1])
        return (lambda: ttnn.add(a_, b_, memory_config=DRAM)), f"{ins[0][0]}"
    if name == "ttnn.reshape":
        x = tensor(*ins[0])
        shape = [int(v) for v in re.search(r"shape = \[([0-9 :i,]+)\]", a).group(1).replace(" : i32", "").split(",")]
        return (lambda: ttnn.reshape(x, shape, memory_config=DRAM)), f"{ins[0][0]} -> {shape}"
    if name == "ttnn.permute":
        x = tensor(*ins[0])
        perm = ints(a, "permutation")
        return (lambda: ttnn.permute(x, perm, memory_config=DRAM)), f"{ins[0][0]} {perm}"
    if name == "ttnn.slice_static":
        x = tensor(*ins[0])
        b_, e_, s_ = (ints_list(a, k) for k in ("begins", "ends", "step"))
        return (lambda: ttnn.slice(x, b_, e_, s_, memory_config=DRAM)), f"{ins[0][0]} [{b_} .. {e_}]"
    if name == "ttnn.concat":
        xs = [tensor(*t) for t in ins]
        dim = one(a, "dim")
        return (lambda: ttnn.concat(xs, dim, memory_config=DRAM)), f"{len(xs)} x {ins[0][0]} dim {dim}"
    if name == "ttnn.to_layout":
        x = tensor(*ins[0])
        shape, dt, tile = outs[0]
        lay = ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT
        return (lambda: ttnn.to_layout(x, lay, dtype=DT[dt], memory_config=DRAM)), \
            f"{ins[0][0]} {ins[0][1]} {'tile' if ins[0][2] else 'rm'} -> {dt} {'tile' if tile else 'rm'}"
    if name == "ttnn.typecast":
        x = tensor(*ins[0])
        return (lambda: ttnn.typecast(x, DT[outs[0][1]], memory_config=DRAM)), f"{ins[0][0]} -> {outs[0][1]}"
    if name == "ttnn.max_pool2d":
        x = tensor(*ins[0])
        kw_args = dict(batch_size=one(a, "batch_size"), input_h=one(a, "input_height"), input_w=one(a, "input_width"),
                       channels=one(a, "channels"), kernel_size=ints(a, "kernel_size"), stride=ints(a, "stride"),
                       padding=ints(a, "padding"), dilation=ints(a, "dilation"), ceil_mode=bool(one(a, "ceil_mode")),
                       memory_config=DRAM, reallocate_halo_output=False)
        return (lambda: ttnn.max_pool2d(x, **kw_args)), f"{ins[0][0]}"
    if name == "ttnn.mean":
        x = tensor(*ins[0])
        dims = [int(v) for v in re.search(r"dim_arg = \[([0-9 :i,]+)\]", a).group(1).replace(" : i32", "").split(",")]
        return (lambda: ttnn.mean(x, dim=dims, keepdim=False, memory_config=DRAM)), f"{ins[0][0]} dims {dims}"
    if name == "ttnn.linear":
        a_, b_ = tensor(*ins[0]), tensor(*ins[1])
        bias = tensor(*ins[2]) if len(ins) > 2 else None
        return (lambda: ttnn.linear(a_, b_, bias=bias, transpose_b=True, memory_config=DRAM)), \
            f"{ins[0][0]} x {ins[1][0]}^T"
    if name == "ttnn.matmul":  # the final average pool as a matmul (AVGPOOL=matmul, S34)
        a_, b_ = tensor(*ins[0]), tensor(*ins[1])
        return (lambda: ttnn.matmul(a_, b_, memory_config=DRAM)), f"{ins[0][0]} x {ins[1][0]}"
    if name == "ttnn.multiply":
        a_, b_ = tensor(*ins[0]), tensor(*ins[1])
        return (lambda: ttnn.multiply(a_, b_, memory_config=DRAM)), f"{ins[0][0]} x {ins[1][0]}"
    return None, "not replayed"


results = []
try:
    for i, o in enumerate(ops):
        r = {"index": i, "op": o["op"], "loc": o["loc"], "in_shapes": [t[0] for t in o["ins"]],
             "out_shapes": [t[0] for t in o["outs"]]}
        try:
            fn, r["desc"] = build(o)
            if fn is None:
                results.append(r)
                print(f"{i:3d} {o['op']:<16} not replayed", flush=True)
                continue
            y = fn()  # compile and first run
            ttnn.synchronize_device(device)
            if isinstance(y, ttnn.Tensor):
                ttnn.deallocate(y)
            new_spans()
            busy, env, host, progs = [], [], [], []
            for _ in range(REPS):
                t0 = time.perf_counter()
                y = fn()
                t1 = time.perf_counter()
                ttnn.synchronize_device(device)
                if isinstance(y, ttnn.Tensor):
                    ttnn.deallocate(y)
                sp, freq = new_spans()
                host.append((t1 - t0) * 1e6)
                busy.append(sum(e - s for s, e in sp) / freq)
                env.append((max(e for _, e in sp) - min(s for s, _ in sp)) / freq if sp else 0.0)
                progs.append(len(sp))
            r.update(device_us=statistics.median(busy), envelope_us=statistics.median(env),
                     host_enqueue_us=statistics.median(host), programs=statistics.median(progs), busy_all=busy)
            print(f"{i:3d} {o['op']:<16} {r['desc']:<48} device {r['device_us']:9.1f} us (envelope "
                  f"{r['envelope_us']:9.1f}, {r['programs']:.0f} programs), host {r['host_enqueue_us']:8.1f} us",
                  flush=True)
        except Exception as e:  # keep going; the op is reported as failed
            r["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            print(f"{i:3d} {o['op']:<16} FAILED {r['error']}", flush=True)
        results.append(r)
finally:
    ttnn.close_device(device)

tot = sum(r.get("device_us", 0.0) for r in results)
print(f"total device time of {sum('device_us' in r for r in results)} replayed ops: {tot / 1e3:.2f} ms", flush=True)
json.dump({"ir": IR, "reps": REPS, "T": T, "ops": results}, open(os.path.expanduser(OUT), "w"), indent=1)
