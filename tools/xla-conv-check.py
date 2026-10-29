# One conv of SEW-ResNet18 (fp32, multi-step input [T, N, C, H, W]) on p150a (tt-xla 1.4.0, torch.compile) vs CPU.
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA140/bin/python -I tools/xla-conv-check.py 1 layer1.0.conv1
# Args: <level> <module path, e.g. conv1 or layer1.0.conv1> [spikes|randn]
import copy, os, sys
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
import torch, torch_xla, torch_xla.runtime as xr
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.model import sew_resnet
level, path = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "spikes"
xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": level})
dev = torch_xla.device()
T, N = 4, 8
torch.manual_seed(20260808)
model = sew_resnet.sew_resnet18(cnf="ADD", num_classes=1000, spiking_neuron=neuron.SimpleLIFNode,
                                tau=2.0, decay_input=True, detach_reset=True, step_mode="m")
functional.set_step_mode(model, "m")
conv = model.get_submodule(path)
cin = conv.in_channels
hw = {3: 224, 64: 56, 128: 28, 256: 14, 512: 7}[cin]
if path.startswith("layer2.0") or path.startswith("layer3.0") or path.startswith("layer4.0"):
    hw *= 2  # first block of a stage takes the previous stage's size
x = (torch.rand(T, N, cin, hw, hw) < 0.03).float() if kind == "spikes" else torch.randn(T, N, cin, hw, hw)
with torch.no_grad():
    ref = conv(x)
m = torch.compile(copy.deepcopy(conv).to(dev), backend="tt")
with torch.no_grad():
    out = m(x.to(dev)); torch_xla.sync(wait=True); out = out.cpu()
pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print(f"level {level} {path} {conv} in {tuple(x.shape)} {kind}: pcc {pcc:.6f} maxabs {(out - ref).abs().max():.4g} "
      f"std dev/cpu {out.std():.4g}/{ref.std():.4g}", flush=True)
