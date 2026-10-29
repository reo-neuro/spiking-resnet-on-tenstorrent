# Does D22 (tt-xla 1.4.0 reads every fp32 CB of the tt-lang LIF rounded to a 10-bit mantissa) account for the drop in
# logits correlation of SEW-ResNet18 with the tt-lang LIF (0.731 against 0.908 for SimpleLIFNode on the device, S18)?
# CPU only. Builds the model as xla-sew18.py does with BN=eval (same seed, input and running statistics), runs it
# with SimpleLIFNode (the reference) and with an LIF that rounds what the kernel reads from CBs (x, h, v) as ttl_lif.py
# computes it, and compares layer by layer (truncation matched the device's tt-lang LIF bit for bit on the device's
# own stem BN output, and the device's SimpleLIFNode matched the exact LIF; S19):
#   isolated: the rounding LIF on the reference's own input to that layer (flips caused by the layer alone)
#   propagated: the rounding LIF's run against the reference run (flips plus everything upstream)
# and the logits against the reference and against device logits saved by xla-sew18.py (SAVE_LOGITS).
# Args: [trunc|rne] [batch] [T] [image_size] [saved_logits.pt ...]
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA140/bin/python -I tools/lif-d22-emul.py trunc 8 4 224 $DATA/logits/*.pt
import json
import sys

import torch
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.model import sew_resnet

HOW = sys.argv[1] if len(sys.argv) > 1 else "trunc"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
H = int(sys.argv[4]) if len(sys.argv) > 4 else 224
SAVED = sys.argv[5:]


def r(a):
    """fp32 -> 10-bit mantissa (TF32), kept in fp32; truncated or rounded to nearest even."""
    b = a.contiguous().view(torch.int32)
    if HOW == "rne":
        b = b + 0x0FFF + ((b >> 13) & 1)
    return (b & -0x2000).view(torch.float32)


class RoundingLIF(torch.nn.Module):
    # ttl_lif.py's arithmetic with every CB read rounded: h0 = x/2, s = (h >= 1), v = h - h*s, h' = v + (x/2 - v/2)
    def forward(self, x_seq):
        xr = r(x_seq)
        h = xr[0] * 0.5
        out = []
        for t in range(x_seq.shape[0]):
            hr = r(h)
            s = (hr >= 1.0).float()
            out.append(s)
            if t + 1 < x_seq.shape[0]:
                v = r(hr - hr * s)
                h = v + (xr[t + 1] * 0.5 - v * 0.5)
        return torch.stack(out)


def exact_h(x_seq):
    """the reference's membrane potential h before firing, per step (SimpleLIFNode, fp32)."""
    v = torch.zeros_like(x_seq[0])
    hs = []
    for xt in x_seq:
        h = v + (xt - v) / 2.0
        hs.append(h)
        v = h * (1.0 - (h >= 1.0).float())
    return torch.stack(hs)


def capture(model, cls):
    """forward hooks that record (input, output) of every neuron of type cls, in call order."""
    rec = []
    for name, mod in model.named_modules():
        if isinstance(mod, cls):
            mod.register_forward_hook(lambda m, i, o, name=name: rec.append((name, i[0].detach(), o.detach())))
    return rec


torch.manual_seed(20260808)
model = sew_resnet.sew_resnet18(
    cnf="ADD", num_classes=1000, spiking_neuron=neuron.SimpleLIFNode,
    tau=2.0, decay_input=True, detach_reset=True, step_mode="m",
)
functional.set_step_mode(model, "m")
x = torch.randn(T, N, 3, H, H)

with torch.no_grad():
    # as xla-sew18.py with BN=eval: running statistics from this batch, then eval mode
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.momentum = None
            mod.reset_running_stats()
    model(x)
    functional.reset_net(model)
    model.eval()

    rec_ref = capture(model, neuron.SimpleLIFNode)
    ref = model(x)
    functional.reset_net(model)

    emul = RoundingLIF()
    for k in [k for k, v in model.named_modules() if isinstance(v, neuron.SimpleLIFNode)]:
        parent, _, leaf = k.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, RoundingLIF())
    rec_emu = capture(model, RoundingLIF)
    out = model(x)


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


layers = []
print(f"rounding: {HOW}, batch {N}, T {T}, {H}x{H}")
print(f"{'layer':<24} {'shape':<22} {'spike rate':>10} {'isolated flips':>15} {'(near)':>7} {'propagated':>11} "
      f"{'input rel err':>13}")
for (name, xi, si), (_, xe, se) in zip(rec_ref, rec_emu):
    iso = emul(xi)
    bad = iso != si
    # near the threshold: |h - 1| < 2^-9 at this step or an earlier one (a flip changes v for the later steps)
    near = ((exact_h(xi) - 1.0).abs() < 2.0**-9).cummax(0).values
    flips_prop = (se != si).float().mean().item()
    rel = ((xe - xi).norm() / xi.norm()).item()
    layers.append({"layer": name, "shape": list(xi.shape), "spike_rate": si.mean().item(),
                   "isolated_flips": bad.float().mean().item(), "isolated_flip_count": int(bad.sum()),
                   "isolated_flip_count_near": int((bad & near).sum()), "propagated_mismatch": flips_prop,
                   "input_rel_err": rel})
    print(f"{name:<24} {str(list(xi.shape)):<22} {si.mean().item():10.4f} {int(bad.sum()):15d} "
          f"{int((bad & near).sum()):7d} {flips_prop:11.2e} {rel:13.2e}")

res = {"rounding": HOW, "batch": N, "T": T, "image_size": H, "layers": layers,
       "logits_pcc_emul_vs_cpu": pcc(out, ref), "logits_maxabs_emul_vs_cpu": (out - ref).abs().max().item()}
print(f"logits: emulated vs CPU pcc {res['logits_pcc_emul_vs_cpu']:.4f}, max abs {res['logits_maxabs_emul_vs_cpu']:.3f}")
saved = {}
for path in SAVED:
    d = torch.load(path, weights_only=True)
    if (d["cpu"] - ref).abs().max().item() != 0.0:
        print(f"{path}: its CPU logits differ from this run's, skipped (max abs {(d['cpu'] - ref).abs().max().item():.3g})")
        continue
    saved[path] = {"device_vs_cpu": pcc(d["device"], ref), "device_vs_emul": pcc(d["device"], out)}
    print(f"{path}: device vs CPU pcc {saved[path]['device_vs_cpu']:.4f}, device vs emulated {saved[path]['device_vs_emul']:.4f}")
res["saved"] = saved
print("RESULT " + json.dumps(res), flush=True)
