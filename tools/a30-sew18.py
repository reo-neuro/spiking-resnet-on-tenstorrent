# A30 baseline for SEW-ResNet18 inference, built like xla-sew18.py (same seed, model and input; BN=eval sets the
# running statistics from the batch as xla-sew18.py does), with the settings that differ between SpikingJelly's
# benchmark (S4) and a30-hostcost.py (S12) as switches, so that they can be changed one at a time:
#   TF32=0|1        cudnn and matmul TF32 (the benchmark turns both off; PyTorch's default allows it for cudnn)
#   CUDNN_BENCH=0|1 torch.backends.cudnn.benchmark (the benchmark leaves it off)
#   BN=eval|train   eval mode as the benchmark's inference phase, or training mode (a30-hostcost.py, D23)
#   GRAD=inference|no_grad   torch.inference_mode() as the benchmark, or torch.no_grad()
# Three timings of one step (forward + reset_net), each the median over the steps:
#   events: back-to-back steps timed with CUDA events, as the benchmark does
#   sync: each step waits for the logits on the host (.cpu()), as xla-sew18.py with WAIT=cpu
#   pipelined: all steps enqueued, then one synchronize, divided by the number of steps (xla-sew18.py's pipelined)
# Args: [LIFNode|SimpleLIFNode] [batch] [T] [image_size] [steps]
# Run: CUDA_VISIBLE_DEVICES=0 TF32=0 CUDNN_BENCH=0 BN=eval $VENV_A30/bin/python -I \
#        tools/a30-sew18.py LIFNode 8 4 224 30
import json
import os
import statistics
import sys
import time

import torch
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.model import sew_resnet

NEURON = sys.argv[1] if len(sys.argv) > 1 else "LIFNode"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
H = int(sys.argv[4]) if len(sys.argv) > 4 else 224
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 30
TF32 = os.environ.get("TF32", "0") == "1"
CUDNN_BENCH = os.environ.get("CUDNN_BENCH", "0") == "1"
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
        # as xla-sew18.py with BN=eval: running statistics from this batch (on CPU, before moving), then eval mode
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


def step():
    out = model(xd)
    functional.reset_net(model)
    return out


with grad():
    for _ in range(10):
        step()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
    for i in range(STEPS):
        starts[i].record()
        step()
        ends[i].record()
    ends[-1].synchronize()
    events = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    sync = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        step().cpu()
        sync.append((time.perf_counter() - t0) * 1e3)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = [step() for _ in range(STEPS)]
    outs[-1].cpu()
    pipelined = (time.perf_counter() - t0) * 1e3 / STEPS

res = {
    "neuron": NEURON, "batch": N, "T": T, "image_size": H, "steps": STEPS, "tf32": TF32, "cudnn_benchmark": CUDNN_BENCH,
    "bn": BN, "grad": GRAD, "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
    "events_median_ms": statistics.median(events), "sync_median_ms": statistics.median(sync),
    "pipelined_ms_per_step": pipelined, "events_ms": events, "sync_ms": sync,
}
print(f"{NEURON} b{N} T{T} {H}: tf32 {int(TF32)} cudnn.benchmark {int(CUDNN_BENCH)} bn {BN} {GRAD}: "
      f"events {res['events_median_ms']:.2f} ms, sync {res['sync_median_ms']:.2f} ms, "
      f"pipelined {pipelined:.2f} ms", flush=True)
print("RESULT " + json.dumps(res), flush=True)
