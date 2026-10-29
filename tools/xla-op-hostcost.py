# Host cost per TTNN op on p150a: chain K copies of one op on a tiny tensor (device work ~0), compile with
# torch.compile(backend="tt"), and time one synced step. The slope of time vs K is the per-op cost.
# Args: <optimization_level> <op> <K...>   op: add | lif | lif_alg | conv | bn | conv_bn_lif
# lif_alg is the same LIF step rewritten with fewer ops (tau=2: (v + x) * 0.5, reset by where).
# OPTS='{"enable_create_d2m_subgraphs": "true"}' adds tt-xla compile options (all values are strings).
# PERF=<file> writes TTNN perf metrics (op counts) for the last K.
# SHAPE=N,C,H,W sets the input (and the conv channels; default 4,16,16,16, where device work is ~0). WAIT=cpu reads
# the output back in every step (torch_xla.sync(wait=True) does not wait for the device, D21), and adds a pipelined
# time (10 steps, one read-back). With real sizes, the slope is the per-op time of the slower of host and device,
# or their sum if the two do not overlap (S23). The pipelined time is split into submit (the host returns from the
# 10 steps) and drain (the read-back after them, i.e. how far the device is behind). REDUCE=1 sums the output over
# all but the batch dimension, so that the read-back is small.
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA140/bin/python -I tools/xla-op-hostcost.py 1 add 50 200
import json, os, statistics, sys, time
os.environ["PJRT_DEVICE"] = "TT"
import torch, torch_xla, torch_xla.runtime as xr

opt, op, Ks = sys.argv[1], sys.argv[2], [int(k) for k in sys.argv[3:]]
xr.set_device_type("TT")
opts = {"optimization_level": opt}
if os.environ.get("OPTS"):
    opts.update(json.loads(os.environ["OPTS"]))
if os.environ.get("PERF"):
    opts["ttnn_perf_metrics_enabled"] = "true"
    opts["ttnn_perf_metrics_output_file"] = os.environ["PERF"]
print("compile options:", opts, flush=True)
torch_xla.set_custom_compile_options(opts)
dev = torch_xla.device()
SHAPE = [int(v) for v in os.environ.get("SHAPE", "4,16,16,16").split(",")]
WAIT_CPU = os.environ.get("WAIT") == "cpu"
REDUCE = os.environ.get("REDUCE") == "1"


class Chain(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        if op in ("conv", "conv_bn_lif"):
            self.convs = torch.nn.ModuleList(torch.nn.Conv2d(SHAPE[1], SHAPE[1], 3, padding=1, bias=False)
                                             for _ in range(k))
        if op in ("bn", "conv_bn_lif"):
            self.bns = torch.nn.ModuleList(torch.nn.BatchNorm2d(SHAPE[1]) for _ in range(k))
            for b in self.bns:
                b.num_batches_tracked = None

    def forward(self, x):
        v = torch.zeros_like(x)
        for i in range(self.k):
            if op == "add":
                x = x + 1.0
            elif op == "lif":  # one SimpleLIFNode step (decay_input, v_reset=0, hard reset, detach_reset)
                v = v + (x - v) / 2.0
                s = (v >= 1.0).to(x.dtype)
                v = v * (1.0 - s)
                x = s
            elif op == "lif_alg":
                v = (v + x) * 0.5
                h = v >= 1.0
                s = h.to(x.dtype)
                v = torch.where(h, 0.0, v)
                x = s
            elif op == "conv":
                x = self.convs[i](x)
            elif op == "bn":
                x = self.bns[i](x)
            elif op == "conv_bn_lif":
                x = self.bns[i](self.convs[i](x))
                v = v + (x - v) / 2.0
                s = (v >= 1.0).to(x.dtype)
                v = v * (1.0 - s)
                x = s
        return x.sum(dim=tuple(range(1, x.dim()))) if REDUCE else x


res = {"optimization_level": opt, "op": op, "points": []}
res["shape"], res["wait"] = SHAPE, "cpu" if WAIT_CPU else "sync"
x = torch.randn(*SHAPE)
with torch.no_grad():
    for k in Ks:
        torch._dynamo.reset()
        m = Chain(k).to(dev)
        mc = torch.compile(m, backend="tt")
        xd = x.to(dev)
        ts = []
        for i in range(12 if WAIT_CPU else 8):
            t0 = time.perf_counter()
            out = mc(xd)
            torch_xla.sync(wait=True)
            if WAIT_CPU:
                out.cpu()
            ts.append((time.perf_counter() - t0) * 1e3)
        med = statistics.median(ts[2:])
        point = {"K": k, "first_ms": ts[0], "median_ms": med}
        if WAIT_CPU:
            t0 = time.perf_counter()
            for i in range(10):
                out = mc(xd)
                torch_xla.sync()
            t1 = time.perf_counter()
            out.cpu()
            t2 = time.perf_counter()
            point["pipelined_ms"] = (t2 - t0) * 1e3 / 10
            point["submit_ms"], point["drain_ms"] = (t1 - t0) * 1e3 / 10, (t2 - t1) * 1e3
        res["points"].append(point)
        print(f"K={k}: first {ts[0]:.0f} ms, median {med:.2f} ms"
              + (f", pipelined {point['pipelined_ms']:.2f} ms (submit {point['submit_ms']:.2f} ms per step, then "
                 f"drain {point['drain_ms']:.1f} ms)" if WAIT_CPU else ""), flush=True)
if len(Ks) >= 2:
    a, b = res["points"][0], res["points"][-1]
    res["per_op_us"] = (b["median_ms"] - a["median_ms"]) / (b["K"] - a["K"]) * 1e3
    msg = f"per op: {res['per_op_us']:.1f} us"
    if WAIT_CPU:
        res["per_op_pipelined_us"] = (b["pipelined_ms"] - a["pipelined_ms"]) / (b["K"] - a["K"]) * 1e3
        msg += f", pipelined {res['per_op_pipelined_us']:.1f} us"
    print(msg, flush=True)
print("RESULT " + json.dumps(res), flush=True)
