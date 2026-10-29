# Time tt-lang LIF kernels (ttl_lif.py and its variants) on p150a, shape by shape, in the tt-lang container (ttnn,
# no tt-xla). Device time per call is the KERNEL span from tt-metal's device profiler (first ZONE_START .. last
# ZONE_END over all cores of one program); host time is call + synchronize.
# Every variant runs on the same input; spikes are compared with the first variant (bit-exact) and with the fp32
# reference. A reference mismatch counts as "near" if |h - 1| < 1e-2 at that position at this or an earlier step
# (a flip near the threshold changes v, so later steps at the same position can differ too).
# Args: <variant>[,<variant>...] [shapes] [reps]
#   variant: <kernel.py>:<factory>:<layout>, factory(T) builds the kernel; layout is how x is handed to it:
#            flat = [1, 1, T*R, Cc] as TtlLIF in xla-sew18.py lays it out, nchw = [T*N, C, H, W] as the BN leaves
#            it, nhwc = [1, 1, T*N*H*W, C] as a conv leaves it ([1, T, N*H*W, C] when N*H*W is not a multiple of
#            32, as in TtlLIF since S23; "split" in the output), split4 = the nhwc rows as four [1, 1, R, C] tensors,
#            one per time step, for make_lif4 (factory() with no T; the stem since S24); split4bf16 = the same
#            with the spikes written to a bf16 out (factory(bf16_out=True), S26); nhwcres = nhwc with a residual r
#            (random integers 0 .. 3, laid out like the output) for make_lif_add, which writes s + r (S30);
#            split2 = the nhwc rows as two [1, 1, 2*R, C] tensors, two time steps each, for make_lif2 (S32)
#   shapes: sew18 (the 20 LIF inputs of SEW-ResNet18 at batch 8, 224x224, T=4, as in the TTNN IR, S16)
#           or B:C:H:W[,B:C:H:W...]  (B = T*N)
# LIF_MEM=l1 puts the input and output tensors in L1 (interleaved) instead of DRAM, l1h in L1 height-sharded over
# the 11x10 grid, as a conv leaves its output (S29: what the LIF costs when the conv before it leaves its output in
# L1); the stem shape does not fit
# Run (inside the tt-lang container; the profiler needs TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal):
#   docker exec -i -e TT_METAL_HOME=/opt/ttlang-toolchain/tt-metal <tt-lang container> bash -c "cd /tmp && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
#     python3 $PWD/tools/ttl-lif-bench.py $PWD/tools/ttl_lif.py:make_lif_nchw:nchw sew18 5"
import importlib.util
import json
import os
import sys
import time

import torch
import ttnn

T = 4
TILE = 32
# (B, C, H, W, how many of the 20 LIF layers have this shape): stem, layer1, layer2, layer3, layer4
SEW18 = [(32, 64, 112, 112, 1), (32, 64, 56, 56, 4), (32, 128, 28, 28, 5), (32, 256, 14, 14, 5), (32, 512, 7, 7, 5)]

specs = [v.rsplit(":", 2) for v in sys.argv[1].split(",")]
spec = sys.argv[2] if len(sys.argv) > 2 else "sew18"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
shapes = SEW18 if spec == "sew18" else [tuple(int(a) for a in s.split(":")) + (1,) for s in spec.split(",")]

variants = []
for i, (path, factory, layout) in enumerate(specs):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    s = importlib.util.spec_from_file_location(f"lif_variant{i}", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    variants.append((f"{os.path.basename(path)}:{factory}", getattr(m, factory), layout, m))
lif_reference = variants[0][3].lif_reference

CSV = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")


def kernel_spans():
    """run host ID -> (start, end, cores) over *-KERNEL zones, and the chip frequency in MHz."""
    st, en, cores = {}, {}, {}
    with open(CSV) as fh:
        freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        fh.readline()
        for line in fh:
            p = line.split(",", 12)
            if not p[10].strip().endswith("-KERNEL"):
                continue
            rid, t = int(p[7]), int(p[5])
            if p[11].strip() == "ZONE_START":
                st[rid] = min(st.get(rid, t), t)
                cores.setdefault(rid, set()).add((int(p[1]), int(p[2])))
            else:
                en[rid] = max(en.get(rid, t), t)
    return {r: (st[r], en[r], len(cores[r])) for r in st if r in en}, freq


def flat_shape(x):
    n = x.numel() // T
    cc = next(c for c in (1024, 512, 256, 128, 64, 32) if n % c == 0 and (n // c) % 32 == 0)
    return (1, 1, T * (n // cc), cc)


def lay_out(x, layout):
    """x as handed to the kernel, a function that turns the kernel output back into x's shape, and the layout."""
    B, C, H, W = x.shape
    if layout in ("split4", "split4bf16", "split2"):
        return (tuple(x.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C).chunk(2 if layout == "split2" else T, dim=2)),
                lambda s: s.reshape(B, H, W, C).permute(0, 3, 1, 2), layout)
    if layout == "nhwcres":
        xin, back, used = lay_out(x, "nhwc")
        return xin, back, "res" if used == "nhwc" else "splitres"
    if layout == "nhwc" and (B // T * H * W) % 32 == 0:
        return x.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C), lambda s: s.reshape(B, H, W, C).permute(0, 3, 1, 2), layout
    if layout == "nhwc":
        return (x.permute(0, 2, 3, 1).reshape(1, T, B // T * H * W, C),
                lambda s: s.reshape(B, H, W, C).permute(0, 3, 1, 2), "split")
    if layout == "flat":
        return x.reshape(flat_shape(x)), lambda s: s.reshape(x.shape), "flat"
    return x, lambda s: s, layout


device = ttnn.open_device(device_id=0)
results = []
try:
    mem = ttnn.L1_MEMORY_CONFIG if os.environ.get("LIF_MEM") == "l1" else ttnn.DRAM_MEMORY_CONFIG
    conv = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    for B, C, H, W, count in shapes:
        if os.environ.get("LIF_MEM") == "l1h":
            rows = -(-B * H * W // TILE // 110) * TILE  # rows per core, whole tiles
            conv["memory_config"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(-(-B * H * W // rows), ttnn.CoreCoord(11, 10), True),
                               (rows, C), ttnn.ShardOrientation.ROW_MAJOR))
        torch.manual_seed(0)
        x = torch.randn(B, C, H, W) * 1.5 + 0.5
        ref, h = lif_reference(x.reshape(T, -1), T)
        r_res = torch.randint(0, 4, x.shape).float()  # the residual of nhwcres
        near = ((h - 1.0).abs() < 1e-2).cummax(0).values
        padded = B * C * -(-H // TILE) * TILE * -(-W // TILE) * TILE
        first = None
        for name, factory, layout, _ in variants:
            xin, back, used = lay_out(x, layout)
            split4 = used.startswith("split4") or used == "split2"  # inputs as a tuple, factory() takes no T
            res = used.endswith("res")
            x_tt = tuple(ttnn.from_torch(g, **conv) for g in xin) if split4 else (ttnn.from_torch(xin, **conv),)
            if res:  # the residual, laid out like x
                x_tt += (ttnn.from_torch(lay_out(r_res, "nhwc")[0], **conv),)
            out_tt = ttnn.from_torch(torch.zeros(1, 1, x.numel() // C, C) if split4 else torch.zeros_like(xin),
                                     **dict(conv, dtype=ttnn.bfloat16 if used == "split4bf16" else ttnn.float32))
            kernel = factory(bf16_out=True) if used == "split4bf16" else factory() if split4 else factory(T)
            t0 = time.perf_counter()
            kernel(*x_tt, out_tt)  # compile + first run
            ttnn.synchronize_device(device)
            first_ms = (time.perf_counter() - t0) * 1e3
            ttnn.ReadDeviceProfiler(device)
            seen = set(kernel_spans()[0])
            host = []
            for _ in range(reps):
                t0 = time.perf_counter()
                kernel(*x_tt, out_tt)
                ttnn.synchronize_device(device)
                host.append((time.perf_counter() - t0) * 1e6)
            ttnn.ReadDeviceProfiler(device)
            spans, freq = kernel_spans()
            new = sorted(r for r in spans if r not in seen)
            dev_us = [(spans[r][1] - spans[r][0]) / freq for r in new]
            s = back(ttnn.to_torch(out_tt).float()).reshape(T, -1)
            if res:
                s = s - r_res.reshape(T, -1)  # s + r - r is exact for these small integers
            bad = s != ref
            if first is None:
                first = s
            dev_med = sorted(dev_us)[len(dev_us) // 2] if dev_us else float("nan")
            obytes = 2 if used == "split4bf16" else 4  # bytes written per element
            moved = (4 + obytes + (4 if res else 0)) * (padded if used == "nchw" else x.numel())
            r = {
                "kernel": name, "layout": layout, "used": used, "shape": [B, C, H, W], "count": count,
                "dev_us": dev_us, "dev_med_us": dev_med, "cores": sorted({spans[r][2] for r in new}),
                "host_us": host, "first_ms": first_ms, "freq_mhz": freq,
                "gbps_logical": (4 + obytes + (4 if res else 0)) * x.numel() / dev_med / 1e3, "gbps_moved": moved / dev_med / 1e3,
                "mismatch": int(bad.sum()), "mismatch_far": int((bad & ~near).sum()),
                "diff_vs_first": int((s != first).sum()),
            }
            results.append(r)
            print(f"{name:>28} {used:>4} [{B},{C},{H},{W}] x{count}: device {dev_med:8.1f} us "
                  f"({r['gbps_logical']:5.1f} GB/s logical, {r['gbps_moved']:5.1f} moved, {r['cores']} cores, "
                  f"{len(dev_us)} runs), host {sorted(host)[len(host) // 2]:8.1f} us, first {first_ms:6.0f} ms, "
                  f"mismatch {r['mismatch']} (far {r['mismatch_far']}), diff vs first {r['diff_vs_first']}",
                  flush=True)
finally:
    ttnn.close_device(device)

for name, _, layout, _ in variants:
    rs = [r for r in results if r["kernel"] == name and r["layout"] == layout]
    tot = sum(r["dev_med_us"] * r["count"] for r in rs)
    print(f"{name} ({layout}): device time of the {sum(r['count'] for r in rs)} LIF calls: {tot / 1e3:.2f} ms",
          flush=True)
print("RESULT " + json.dumps(results), flush=True)
