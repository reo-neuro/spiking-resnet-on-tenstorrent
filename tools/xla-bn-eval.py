# Conv -> eval-mode BatchNorm2d forward on p150a, lazy tensor vs torch.compile(backend="tt"),
# following tt-xla tests/benchmark (num_batches_tracked=None). Compare with CPU.
# Args: [lazy|compile] [optimization_level]
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA/bin/python tools/xla-bn-eval.py compile 0
import copy
import os
import sys
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
import torch
from torch import nn
import torch_xla
import torch_xla.runtime as xr

mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
level = sys.argv[2] if len(sys.argv) > 2 else "0"
xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": level})
dev = torch_xla.device()

torch.manual_seed(0)
net = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
with torch.no_grad():  # non-trivial running stats
    net[1].running_mean.uniform_(-1, 1)
    net[1].running_var.uniform_(0.5, 2)
    net[1].weight.uniform_(0.5, 1.5)
    net[1].bias.uniform_(-0.5, 0.5)
net.eval()
x = torch.randn(8, 16, 28, 28)
with torch.no_grad():
    ref = net(x)

m = copy.deepcopy(net)
m[1].num_batches_tracked = None
if mode == "compile":
    m.compile(backend="tt")
m = m.to(dev)
with torch.no_grad():
    out = m(x.to(dev))
    torch_xla.sync(wait=True)
out = out.cpu()
pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print(f"{mode} level={level}: maxabs={(out - ref).abs().max().item():.3g} pcc={pcc:.6f}", flush=True)
