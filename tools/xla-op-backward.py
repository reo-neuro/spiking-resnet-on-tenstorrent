# Isolate which op's backward differs between p150a (torch_xla) and CPU.
# Each case: random input and upstream grad, compare output, input grad and parameter grads.
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA/bin/python tools/xla-op-backward.py
import copy
import os
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
import torch
from torch import nn
import torch_xla
import torch_xla.runtime as xr
from sj_xla import ManualBN2d

xr.set_device_type("TT")
dev = torch_xla.device()
# OPTS='{"fp32_dest_acc_en": "true"}' passes tt-xla compile options (all values are strings)
if os.environ.get("OPTS"):
    import json
    torch_xla.set_custom_compile_options(json.loads(os.environ["OPTS"]))
    print("compile options:", os.environ["OPTS"], flush=True)
torch.manual_seed(0)

cases = {
    "conv1": (nn.Conv2d(1, 16, 3, padding=1, bias=False), (16, 1, 28, 28)),
    "conv2": (nn.Conv2d(16, 32, 3, padding=1, bias=False), (16, 16, 14, 14)),
    "bn_train": (nn.BatchNorm2d(16), (16, 16, 28, 28)),
    "bn_eval": (nn.BatchNorm2d(16).eval(), (16, 16, 28, 28)),
    "maxpool": (nn.MaxPool2d(2), (16, 16, 28, 28)),
    "relu": (nn.ReLU(), (16, 16, 28, 28)),
    "linear": (nn.Linear(1568, 10, bias=False), (16, 1568)),
    "conv_bn": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16)), (16, 1, 28, 28)),
    "conv_bn_relu": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU()), (16, 1, 28, 28)),
    "conv_bn_relu_pool": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)), (16, 1, 28, 28)),
    "conv_relu_pool": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.ReLU(), nn.MaxPool2d(2)), (16, 1, 28, 28)),
    "conv_manualbn": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), ManualBN2d(16)), (16, 1, 28, 28)),
    "conv_gn": (nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16)), (16, 1, 28, 28)),
    "full_relu_T1": (nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10, bias=False)), (16, 1, 28, 28)),
}
only = os.environ.get("ONLY")
if only:
    cases = {k: v for k, v in cases.items() if k in only.split(",")}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def run(m0, x, g, device):
    m = copy.deepcopy(m0).to(device)
    xi = x.detach().clone().to(device).requires_grad_()
    out = m(xi)
    out.backward(g.to(device))
    if device != "cpu":
        torch_xla.sync(wait=True)
    return out.detach().cpu(), xi.grad.cpu(), [p.grad.cpu() for p in m.parameters()]


for name, (m0, shape) in cases.items():
    x = torch.randn(shape)
    with torch.no_grad():
        g = torch.randn_like(m0(x))
    try:
        o_c, gx_c, gp_c = run(m0, x, g, "cpu")
        o_t, gx_t, gp_t = run(m0, x, g, dev)
    except Exception as e:  # report and continue with the next op
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        continue
    line = (f"{name}: out maxabs={(o_c - o_t).abs().max().item():.3g} pcc={pcc(o_c, o_t):.6f} | "
            f"grad_x maxabs={(gx_c - gx_t).abs().max().item():.3g} pcc={pcc(gx_c, gx_t):.6f} "
            f"(cpu max {gx_c.abs().max().item():.3g}, tt max {gx_t.abs().max().item():.3g})")
    for i, (a, b) in enumerate(zip(gp_c, gp_t)):
        line += (f" | grad_p{i} maxabs={(a - b).abs().max().item():.3g} pcc={pcc(a, b):.6f}"
                 f" (cpu max {a.abs().max().item():.3g}, tt max {b.abs().max().item():.3g})")
    print(line, flush=True)
