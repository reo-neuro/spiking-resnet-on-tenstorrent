# The stem conv (3->64, 7x7, stride 2, pad 3) at level 1 with the input channels zero-padded to cin (weights too).
# Args: <level> <cin> [batch]
# tt-xla 1.4.0 at level 1 gets 3, 4, 8 and 16 channels wrong and 32 right (S18, D24). Run like xla-conv-check.py.
import copy, os, sys
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
import torch, torch_xla, torch_xla.runtime as xr
level, cin = sys.argv[1], int(sys.argv[2])
B = int(sys.argv[3]) if len(sys.argv) > 3 else 32
xr.set_device_type("TT")
torch_xla.set_custom_compile_options({"optimization_level": level})
dev = torch_xla.device()
torch.manual_seed(0)
conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
x = torch.randn(B, 3, 224, 224)
with torch.no_grad():
    ref = conv(x)
    pc = torch.nn.Conv2d(cin, 64, 7, stride=2, padding=3, bias=False)
    pc.weight.zero_()
    pc.weight[:, :3] = conv.weight
    xp = torch.zeros(B, cin, 224, 224)
    xp[:, :3] = x
m = torch.compile(pc.to(dev), backend="tt")
with torch.no_grad():
    out = m(xp.to(dev)); torch_xla.sync(wait=True); out = out.cpu()
pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print(f"level {level} cin {cin} batch {B}: pcc {pcc:.6f} maxabs {(out - ref).abs().max():.4g} std dev/cpu {out.std():.4g}/{ref.std():.4g}", flush=True)
