# Per-layer GPU time of SEW-ResNet18 inference on the A30, built like a30-sew18.py (and xla-sew18.py): torch.profiler
# with a record_function range around every leaf layer (conv, BN, neuron, pooling, fc), so each range's device time
# is the kernels launched inside it; kernels outside every range (the SEW residual adds, copies) are "other". Also
# the host time of one forward call (returns before the GPU finishes) and the synced step time without the profiler.
# Same switches as a30-sew18.py: TF32, CUDNN_BENCH, BN, GRAD.
# Args: [LIFNode|SimpleLIFNode] [batch] [T] [image_size] [profiled steps] [out.json]
# Run: CUDA_VISIBLE_DEVICES=0 TF32=1 CUDNN_BENCH=1 $VENV_A30/bin/python -I \
#        tools/a30-sew18-profile.py LIFNode 8 4 224 10 $DATA/a30-profile.json
import json
import os
import statistics
import sys
import time

import torch
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.model import sew_resnet
from torch.profiler import ProfilerActivity, profile

NEURON = sys.argv[1] if len(sys.argv) > 1 else "LIFNode"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
H = int(sys.argv[4]) if len(sys.argv) > 4 else 224
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
OUT = sys.argv[6] if len(sys.argv) > 6 else ""
TF32 = os.environ.get("TF32", "1") == "1"
CUDNN_BENCH = os.environ.get("CUDNN_BENCH", "1") == "1"
BN = os.environ.get("BN", "eval")
GRAD = os.environ.get("GRAD", "inference")

torch.backends.cuda.matmul.allow_tf32 = TF32
torch.backends.cudnn.allow_tf32 = TF32
torch.backends.cudnn.benchmark = CUDNN_BENCH
dev = torch.device("cuda")

torch.manual_seed(20260808)
model = sew_resnet.sew_resnet18(
    cnf="ADD", num_classes=1000, spiking_neuron=getattr(neuron, NEURON),
    tau=2.0, decay_input=True, detach_reset=True, step_mode="m",
)
functional.set_step_mode(model, "m")
x = torch.randn(T, N, 3, H, H)
with torch.no_grad():
    if BN == "eval":
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momentum = None
                mod.reset_running_stats()
        model(x)
        functional.reset_net(model)
model.train(BN != "eval")
model = model.to(dev)
xd = x.to(dev)
grad = torch.inference_mode if GRAD == "inference" else torch.no_grad

# a record_function range around every layer of these types, named "L:<module path>|<class>" (neurons are not leaf
# modules: they hold their surrogate function as a child)
LAYER_TYPES = (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.Linear,
               neuron.BaseNode, neuron.SimpleBaseNode)
leaves = [(n, m) for n, m in model.named_modules() if n and isinstance(m, LAYER_TYPES)]
handles = []
for name, mod in leaves:
    tag = f"L:{name}|{type(mod).__name__}"

    def pre(m, i, tag=tag):
        m._rf = torch.ops.profiler._record_function_enter_new(tag, None)

    def post(m, i, o):
        torch.ops.profiler._record_function_exit._RecordFunction(m._rf)

    handles.append(mod.register_forward_pre_hook(pre))
    handles.append(mod.register_forward_hook(post))


def step():
    out = model(xd)
    functional.reset_net(model)
    return out


with grad():
    for _ in range(40):  # past the slow phase seen with TF32 (S20)
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
    for h in handles:
        h.remove()
    calls, synced = [], []
    for _ in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = step()
        t1 = time.perf_counter()
        out.cpu()
        t2 = time.perf_counter()
        calls.append((t1 - t0) * 1e3)
        synced.append((t2 - t0) * 1e3)

per_layer, kernel_total = {}, 0.0
for e in prof.events():
    # the CPU-side range (its device time is the kernels launched inside); the range also shows up as a GPU-side
    # annotation of the same name, which is skipped here
    if e.device_type != torch.autograd.DeviceType.CPU or not e.name.startswith("L:"):
        continue
    dt = getattr(e, "device_time_total", None)
    if dt is None:
        dt = e.cuda_time_total
    if True:
        per_layer[e.name[2:]] = per_layer.get(e.name[2:], 0.0) + dt
# all GPU kernels in the profiled window (key_averages also lists CPU ops; keep the CUDA ones)
cuda_events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.key.startswith("L:")]
self_dev = lambda e: getattr(e, "self_device_time_total", None) or e.self_cuda_time_total
kernel_total = sum(self_dev(e) for e in cuda_events)
layers = [{"layer": k.split("|")[0], "type": k.split("|")[1], "gpu_us": v / STEPS} for k, v in per_layer.items()]
order = {n: i for i, (n, _) in enumerate(leaves)}
layers.sort(key=lambda r: order.get(r["layer"], 1e9))
in_layers = sum(r["gpu_us"] for r in layers)
top_kernels = sorted(((e.key, self_dev(e) / STEPS) for e in cuda_events), key=lambda kv: -kv[1])[:25]
res = {
    "neuron": NEURON, "batch": N, "T": T, "image_size": H, "tf32": TF32, "cudnn_benchmark": CUDNN_BENCH, "bn": BN,
    "grad": GRAD, "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "profiled_steps": STEPS,
    "layers": layers, "gpu_in_layers_us": in_layers, "gpu_kernels_total_us": kernel_total / STEPS,
    "host_call_ms": statistics.median(calls), "synced_step_ms": statistics.median(synced),
    "top_kernels_us": top_kernels,
}
by_type = {}
for r in layers:
    by_type[r["type"]] = by_type.get(r["type"], 0.0) + r["gpu_us"]
print(f"{NEURON} b{N} tf32 {int(TF32)} cudnn.benchmark {int(CUDNN_BENCH)} bn {BN}: synced step {res['synced_step_ms']:.2f} ms, "
      f"host call {res['host_call_ms']:.2f} ms, GPU in layers {in_layers / 1e3:.2f} ms, all kernels "
      f"{res['gpu_kernels_total_us'] / 1e3:.2f} ms", flush=True)
print("by type (ms): " + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])))
for k, v in top_kernels[:12]:
    print(f"  {v:8.1f} us  {k[:110]}")
if OUT:
    json.dump(res, open(os.path.expanduser(OUT), "w"), indent=1)
print("RESULT " + json.dumps({k: v for k, v in res.items() if k not in ("layers", "top_kernels_us")}), flush=True)
