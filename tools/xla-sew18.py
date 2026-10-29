# SEW-ResNet18 inference on p150a via torch_xla, built like SpikingJelly's
# benchmark/benchmark_snn_single_gpu.py (cnf=ADD, tau=2, detach_reset, step_mode m,
# random input [T, N, 3, H, W], fp32) but with SimpleLIFNode instead of LIFNode (LIFNode has no XLA kernel).
# The benchmark's inference phase runs the model in eval mode; use BN=eval to match it (see below).
# Args: [compile|lazy] [batch] [T] [steps] [image_size] [optimization_level] [trace|notrace] [perf_metrics_file]
# trace follows tt-xla docs/source/performance.md: TT_RUNTIME_TRACE_REGION_SIZE before importing torch_xla, and the
# option set globally so every TT program in the process is traced (mixing traced and untraced ones broke buffers,
# overall log 52-53). The second run captures the trace, so it is excluded from the median along with the first.
# NEURON=ttl swaps every SimpleLIFNode on the device model for the tt-lang LIF kernel (ttl_lif.py) called through
# tt-xla's @tt_lang_operation (plugin 1.4.0 / venv-xla140); the CPU reference keeps SimpleLIFNode.
# EXPORT=<dir> dumps the intermediate IRs (tt-xla's export_path).
# PAUSE=<step>:<seconds>[:spin] sleeps (or busy-loops on the host, without device work) before that step, to see
# whether a slowdown is a transient state and whether host CPU activity alone keeps it.
# LIF_LAYOUT=nhwc|nchw (with NEURON=ttl) calls the kernel on [T*N*H*W, C] or on [T*N, C, H, W] instead of
# [T*R, Cc] (see TtlLIF).
# LIF_RES=1 (with NEURON=ttl LIF_LAYOUT=nhwc4) folds each SEW block's residual add into its second LIF
# (make_lif_add writes s + identity; S30).
# LIF_OUT=zeros hands every tt-lang LIF a torch.zeros_like output buffer, which tt-mlir merges per shape (D25; for
# reproducing it only).
# NEURON=none replaces them with identity, on the device and in the CPU reference (to time everything but the
# neurons, and to see the precision of the rest without spike thresholds). SAVE_LOGITS=<file.pt> saves the first
# step's device logits and the CPU reference.
# NEURON=reshape keeps the [T*R, Cc] layout change of NEURON=ttl but puts x * 0.5 in place of the kernel (to time
# the layout change; logits differ from CPU).
# BN=eval runs the model in eval mode, as SpikingJelly's benchmark does for inference (model.train(False)); the
# running statistics are first set from the input batch so that the network sees realistic activations, and the CPU
# reference runs in eval mode too. tt-mlir folds the eval-mode BN into the convs. Without BN=eval the BN runs in
# training mode (S4-S16, a misreading of the benchmark: D23). BN=none replaces the BNs with identity.
# WAIT=cpu reads the logits back to the host in every step: torch_xla.sync(wait=True) does not wait for the device
# (S16), so without it a step time is the host's enqueue time until the device falls behind.
# STEM_FOLD=1 (with STEM_PAD) folds the input 2x2 into the channels on the host and runs the stem as a 4x4 stride-1
# conv (see STEM_PAD below).
# STEM_SPLIT=k (with BN=eval) runs the stem conv as k convs over consecutive groups of the T*N images, its BN
# folded in on the host, and concatenates their outputs (see STEM_SPLIT below). STEM_LIF4=1 (with STEM_SPLIT=4 or 2,
# T=4, NEURON=ttl) has the stem LIF read the four (make_lif4) or two (make_lif2, S32) outputs instead of the concat.
# STEM_SPIKE=bf16 (with STEM_LIF4, STEM_SPLIT=4) has that LIF write its spikes (exactly 0 or 1) in bf16, the dtype
# the stem's maxpool runs in, and casts the maxpool output back to fp32 (S26).
# INPUT_LAYOUT=nhwc puts the input on the device as [1, 1, T*N*H*W, C] (laid out on the host) instead of
# [T, N, C, H, W]; the model sees the same NCHW tensor. With STEM_SPLIT=k it is k inputs [1, 1, T*N*H*W/k, C], one
# per image group, that go to the split stem as they are.
# AVGPOOL=matmul computes the final average pool as a matmul with a 0/1 matrix (S34).
# PREFIX=<stage> (conv1, bn1, sn1, maxpool, layer1 .. layer4) runs the model only up to that stage, on the device and
# in the CPU reference, and compares its output instead of the logits (mismatching elements, relative error).
# Run: LD_LIBRARY_PATH=$UV_PY312_LIB \
#      $VENV_XLA/bin/python tools/xla-sew18.py compile 8 4 20
import copy
import json
import os
import statistics
import sys
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
TRACE = len(sys.argv) > 7 and sys.argv[7] == "trace"
if TRACE:
    os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = "10000000"
import resource
import time
import torch
import torch_xla
import torch_xla.runtime as xr
from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based.model import sew_resnet

mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
H = int(sys.argv[5]) if len(sys.argv) > 5 else 224
OPT = sys.argv[6] if len(sys.argv) > 6 else "0"
PERF = sys.argv[8] if len(sys.argv) > 8 else ""
xr.set_device_type("TT")
opts = {"optimization_level": OPT}
if TRACE:
    opts["enable_trace"] = "true"
if os.environ.get("EXPORT"):  # dump the intermediate IRs (TTIR, TTNN) into this directory
    opts["export_path"] = os.environ["EXPORT"]
    opts["export_model_name"] = "sew18"
if PERF:
    opts["ttnn_perf_metrics_enabled"] = "true"
    opts["ttnn_perf_metrics_output_file"] = PERF
torch_xla.set_custom_compile_options(opts)
dev = torch_xla.device()

torch.manual_seed(20260808)
model = sew_resnet.sew_resnet18(
    cnf="ADD", num_classes=1000, spiking_neuron=neuron.SimpleLIFNode,
    tau=2.0, decay_input=True, detach_reset=True, step_mode="m",
)
functional.set_step_mode(model, "m")
x = torch.randn(T, N, 3, H, H)
PREFIX = os.environ.get("PREFIX")
STAGES = ("conv1", "bn1", "sn1", "maxpool", "layer1", "layer2", "layer3", "layer4")


class Prefix(torch.nn.Module):
    # the stages of SEWResNet._forward_impl up to and including `last`
    def __init__(self, net, last):
        super().__init__()
        self.stages = torch.nn.Sequential(*(getattr(net, s) for s in STAGES[: STAGES.index(last) + 1]))

    def forward(self, x):
        return self.stages(x)


with torch.no_grad():
    if os.environ.get("BN") == "eval":
        # SpikingJelly's benchmark runs inference with model.train(False) (S17). Copy this batch's statistics into the
        # running statistics first (momentum=None is a plain average, here over one batch), so that the eval-mode
        # network sees realistic activations; the reference and the device model then both run in eval mode.
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.momentum = None
                mod.reset_running_stats()
        model(x)
        functional.reset_net(model)
        model.eval()
    if os.environ.get("NEURON") == "none":  # compare like with like: identity neurons on the CPU too
        for k in [k for k, v in model.named_modules() if isinstance(v, neuron.SimpleLIFNode)]:
            parent, _, leaf = k.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, torch.nn.Identity())
    ref = model(x)
    if PREFIX:
        functional.reset_net(model)
        ref = Prefix(model, PREFIX)(x)
functional.reset_net(model)

m = copy.deepcopy(model)
LIF_LAYOUT = os.environ.get("LIF_LAYOUT", "flat")
if os.environ.get("NEURON") == "ttl":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tt_torch  # noqa: F401
    from tt_torch.tt_lang import tt_lang_operation
    from ttl_lif import make_lif, make_lif2, make_lif4, make_lif_add, make_lif_nchw

    # LIF_GRID=cols,rows: the grid of make_lif and make_lif4. tt-xla resolves a kernel against a stub 8x8 Wormhole
    # device, so grid="auto" gives 64 cores; the p150a has 11x10 (S26).
    LIF_GRID = tuple(int(a) for a in os.environ["LIF_GRID"].split(",")) if os.environ.get("LIF_GRID") else "auto"

    # A fresh kernel per resolve: tt-lang caches per kernel object, and a cache hit bypasses the compile
    # hook that tt_torch.tt_lang uses to capture the kernel.
    @tt_lang_operation(operation_id=f"snn.lif.t{T}.v1", arg_roles=("in", "out"), version_tag="s13-v1")
    def lif_op(x, out):
        return make_lif(T, grid=LIF_GRID)(x, out)

    @tt_lang_operation(operation_id=f"snn.lif_add.t{T}.v1", arg_roles=("in", "in", "out"), version_tag="s30-v1")
    def lif_add_op(x, r, out):
        return make_lif_add(T, grid=LIF_GRID)(x, r, out)

    @tt_lang_operation(operation_id=f"snn.lif_nchw.t{T}.v1", arg_roles=("in", "out"), version_tag="s16-v1")
    def lif_nchw_op(x, out):
        return make_lif_nchw(T)(x, out)

    @tt_lang_operation(operation_id="snn.lif4.v1", arg_roles=("in", "in", "in", "in", "out"), version_tag="s24-v1")
    def lif4_op(x0, x1, x2, x3, out):
        return make_lif4(grid=LIF_GRID)(x0, x1, x2, x3, out)

    @tt_lang_operation(operation_id="snn.lif2.v1", arg_roles=("in", "in", "out"), version_tag="s32-v1")
    def lif2_op(x0, x1, out):
        return make_lif2(grid=LIF_GRID)(x0, x1, out)

    @tt_lang_operation(operation_id="snn.lif4.bf16.v1", arg_roles=("in", "in", "in", "in", "out"), version_tag="s26-v1")
    def lif4_bf16_op(x0, x1, x2, x3, out):
        return make_lif4(bf16_out=True, grid=LIF_GRID)(x0, x1, x2, x3, out)

    class TtlLIF(torch.nn.Module):
        # multi-step LIF on [T, ...]: lay the rest out as [1, 1, T*R, Cc] tiles and run all T steps in one kernel.
        # LIF_LAYOUT=nhwc lays [T, N, C, H, W] out as [1, 1, T*N*H*W, C], the layout the convolutions use, when N*H*W
        # and C are multiples of 32 (so each time step starts on a tile row); other shapes keep [T*R, Cc]. It adds
        # permutes instead of removing reshapes (S16). LIF_LAYOUT=nchw hands the kernel [T*N, C, H, W] as is.
        # The kernel writes into its "out" operand in place (ttnn.generic), and tt-mlir merges equal constants, so
        # with torch.zeros_like every LIF of one shape wrote into one buffer and a block's ADD of two LIF outputs
        # read the later one twice (D25). Each LIF fills its out with its own value instead (the kernel overwrites
        # every element); LIF_OUT=zeros brings the shared buffer back.
        def __init__(self, k):
            super().__init__()
            self.fill = 0.0 if os.environ.get("LIF_OUT") == "zeros" else -1.0 - k

        def forward(self, x, r=None):
            # r: the residual of a SEW block (LIF_RES=1), added to the spikes inside the kernel (nhwc4 only)
            assert r is None or LIF_LAYOUT == "nhwc4", "LIF_RES=1 needs LIF_LAYOUT=nhwc4"
            if LIF_LAYOUT == "nchw" and x.dim() == 5:
                x2 = x.flatten(0, 1)
                return lif_nchw_op(x2, torch.full_like(x2, self.fill)).reshape(x.shape)
            if LIF_LAYOUT == "nhwc4" and x.dim() == 5:
                # permute the 4D [T*N, C, H, W] view, so that the permute sits right next to the one tt-mlir puts
                # after each conv (NHWC -> NCHW) and the pair can fold away
                # when N*H*W is not a multiple of 32 (layer4), hand [1, T, N*H*W, C] so that each time step starts
                # on a tile (S23); [1, 1, T*N*H*W, C] there would go through the generic path below, a 7x7 NCHW
                t_, n_, c_, h_, w_ = x.shape
                if c_ % 32 == 0:
                    a_ = 1 if (n_ * h_ * w_) % 32 == 0 else t_
                    def lay(z):
                        return z.reshape(t_ * n_, c_, h_, w_).permute(0, 2, 3, 1).reshape(1, a_, t_ * n_ * h_ * w_ // a_, c_)

                    x2 = lay(x)
                    if r is None:
                        y = lif_op(x2, torch.full_like(x2, self.fill))
                    else:  # the residual comes in the NCHW view too; permuting it back folds like x's permute
                        y = lif_add_op(x2, lay(r), torch.full_like(x2, self.fill))
                    y = y.reshape(t_ * n_, h_, w_, c_).permute(0, 3, 1, 2).reshape(x.shape)
                    # the last LIF feeds the avgpool, whose (1, 1) case on a channels-last-strided input ends in an
                    # as_strided_ that torch_xla rejects ("Input tensor is not an XLA tensor")
                    return y.contiguous() if a_ > 1 else y
            if LIF_LAYOUT == "nhwc" and x.dim() == 5:
                t_, n_, c_, h_, w_ = x.shape
                if (n_ * h_ * w_) % 32 == 0 and c_ % 32 == 0:
                    x2 = x.permute(0, 1, 3, 4, 2).reshape(1, 1, t_ * n_ * h_ * w_, c_)
                    y = lif_op(x2, torch.full_like(x2, self.fill))
                    return y.reshape(t_, n_, h_, w_, c_).permute(0, 1, 4, 2, 3)
            n = x[0].numel()
            cc = next(c for c in (1024, 512, 256, 128, 64, 32) if n % c == 0 and (n // c) % 32 == 0)
            x2 = x.reshape(1, 1, T * (n // cc), cc)
            return lif_op(x2, torch.full_like(x2, self.fill)).reshape(x.shape)

    names = [k for k, v in m.named_modules() if isinstance(v, neuron.SimpleLIFNode)]
    for i, k in enumerate(names):
        parent, _, leaf = k.rpartition(".")
        setattr(m.get_submodule(parent) if parent else m, leaf, TtlLIF(i))
    print(f"replaced {len(names)} SimpleLIFNode with the tt-lang LIF kernel", flush=True)
    if os.environ.get("LIF_RES") == "1":
        class ResLIFBlock(torch.nn.Module):
            # SEW BasicBlock.forward with cnf="ADD", the add done by sn2's kernel: identity + sn2(bn2(conv2(...)))
            # (the identity is computed first so that the kernel can read it; the same values in the same order)
            def __init__(self, blk):
                super().__init__()
                assert blk.cnf == "ADD"
                self.blk = blk

            def forward(self, x):
                b = self.blk
                out = b.bn2(b.conv2(b.sn1(b.bn1(b.conv1(x)))))
                identity = b.downsample_sn(b.downsample(x)) if b.downsample is not None else x
                return b.sn2(out, identity)

        nb = 0
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            seq = getattr(m, lname)
            for j in range(len(seq)):
                seq[j] = ResLIFBlock(seq[j])
                nb += 1
        print(f"folded the residual add of {nb} SEW blocks into their second LIF", flush=True)
elif os.environ.get("NEURON") == "reshape":
    class ReshapeOnly(torch.nn.Module):
        # same reshapes as TtlLIF, with one elementwise op between them so that they do not fold away
        def forward(self, x):
            n = x[0].numel()
            cc = next(c for c in (1024, 512, 256, 128, 64, 32) if n % c == 0 and (n // c) % 32 == 0)
            return (x.reshape(T * (n // cc), cc) * 0.5).reshape(x.shape)

    names = [k for k, v in m.named_modules() if isinstance(v, neuron.SimpleLIFNode)]
    for k in names:
        parent, _, leaf = k.rpartition(".")
        setattr(m.get_submodule(parent) if parent else m, leaf, ReshapeOnly())
    print(f"replaced {len(names)} SimpleLIFNode with reshape * 0.5 reshape", flush=True)
elif os.environ.get("NEURON") == "none":
    names = [k for k, v in m.named_modules() if isinstance(v, neuron.SimpleLIFNode)]
    for k in names:
        parent, _, leaf = k.rpartition(".")
        setattr(m.get_submodule(parent) if parent else m, leaf, torch.nn.Identity())
    print(f"replaced {len(names)} SimpleLIFNode with identity", flush=True)
for mod in m.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.num_batches_tracked = None
if os.environ.get("BN") == "none":
    names = [k for k, v in m.named_modules() if isinstance(v, torch.nn.BatchNorm2d)]
    for k in names:
        parent, _, leaf = k.rpartition(".")
        setattr(m.get_submodule(parent) if parent else m, leaf, torch.nn.Identity())
    print(f"replaced {len(names)} BatchNorm2d with identity", flush=True)
xin = x
STEM_PAD = int(os.environ.get("STEM_PAD", "0"))
if STEM_PAD:
    # tt-xla 1.4.0 at optimization_level 1 computes the 3-channel stem conv wrongly (correlation 0.28 with CPU; 4, 8
    # and 16 channels too, 32 is right; D24). Zero-pad its input channels (and weights) on the device model only.
    # STEM_FOLD=1 first rewrites the 7x7 stride-2 conv as a 4x4 stride-1 conv on the input folded 2x2 into the
    # channels (space-to-depth, done on the host: pad 3 -> 230x230 -> [12, 115, 115], channel c*4 + p*2 + q; the
    # kernel is padded to 8x8 and folded the same way), as tt-metal's ResNet-50 does with its stem: a quarter of the
    # input data and a third of the multiply-adds (S23). The same sum, in another order.
    fold = os.environ.get("STEM_FOLD") == "1"
    with torch.no_grad():
        old = m.conv1
        w, xs = old.weight, x
        if fold:
            p_ = old.padding[0]
            xs = torch.nn.functional.pad(x, (p_, p_, p_, p_))
            hh = xs.shape[-1] // 2
            xs = xs.reshape(T, N, 3, hh, 2, hh, 2).permute(0, 1, 2, 4, 6, 3, 5).reshape(T, N, 12, hh, hh)
            k8 = old.kernel_size[0] + 1
            w = torch.nn.functional.pad(w, (0, 1, 0, 1)).reshape(old.out_channels, 3, k8 // 2, 2, k8 // 2, 2)
            w = w.permute(0, 1, 3, 5, 2, 4).reshape(old.out_channels, 12, k8 // 2, k8 // 2)
        pad = type(old)(STEM_PAD, old.out_channels, w.shape[-1], stride=1 if fold else old.stride,
                        padding=0 if fold else old.padding, bias=False, step_mode="m")
        pad.weight.zero_()
        pad.weight[:, : w.shape[1]] = w
        m.conv1 = pad
        xin = torch.zeros(T, N, STEM_PAD, *xs.shape[-2:])
        xin[:, :, : xs.shape[2]] = xs
    print(f"stem conv input padded to {STEM_PAD} channels" + (" after folding 2x2 into 12" if fold else ""),
          flush=True)
STEM_SPLIT = int(os.environ.get("STEM_SPLIT", "1"))
if STEM_SPLIT > 1:
    # Run the stem conv as STEM_SPLIT convs over consecutive groups of the T*N images (S24). On all 32 images its
    # output does not fit L1, so tt-mlir slices it through DRAM (3.8 ms, 22 programs); a group of 8 or 16 runs from
    # L1 in about 1.0 ms in all (ttnn-stem-conv-bench.py). The eval-mode BN is folded into the conv on the host
    # (tt-mlir would fold it into each conv, but not across the concat), and each output is laid out as the
    # [1, 1, b*H*W, C] rows tt-mlir's conv writes, so that the permutes fold away and the concat is along rows.
    # With INPUT_LAYOUT=nhwc the input comes as k tensors [1, 1, b*H*W, C], one per group (see there): slicing the
    # [T*N, C, H, W] view instead left a reshape to [T*N, H, W, C] and back around each slice, and with H = 115 not a
    # multiple of 32 those move the data (1.4 ms); slicing [1, k, b*H*W, C] along dim 1 still copied (0.28 ms).
    assert os.environ.get("BN") == "eval", "STEM_SPLIT folds the eval-mode BN into the stem conv"
    assert T * N % STEM_SPLIT == 0, "STEM_SPLIT must divide T*N"

    class SplitStem(torch.nn.Module):
        def __init__(self, conv, bn, k, shape):
            super().__init__()
            w, b = torch.nn.utils.fusion.fuse_conv_bn_weights(conv.weight, conv.bias, bn.running_mean, bn.running_var,
                                                              bn.eps, bn.weight, bn.bias)
            self.weight, self.bias = torch.nn.Parameter(w.detach()), torch.nn.Parameter(b.detach())
            self.stride, self.padding, self.k, self.shape = conv.stride, conv.padding, k, shape
            self.lif_fill = None  # set for STEM_LIF4: the fill of the LIF's out buffer
            self.spike_bf16 = False  # STEM_SPIKE=bf16: the LIF writes bf16 spikes

        def forward(self, x):
            t_, n_, ci, hi, wi = self.shape
            if isinstance(x, tuple):  # k tensors [1, 1, b*H*W, C] (INPUT_LAYOUT=nhwc)
                gs = [g.reshape(-1, hi, wi, ci).permute(0, 3, 1, 2) for g in x]
            else:
                gs = x.flatten(0, 1).chunk(self.k)
            ys = [torch.nn.functional.conv2d(g, self.weight, self.bias, self.stride, self.padding) for g in gs]
            c_, h_, w_ = ys[0].shape[1:]
            ys = [g.permute(0, 2, 3, 1).reshape(1, 1, -1, c_) for g in ys]
            if self.lif_fill is None:
                y = torch.cat(ys, dim=2)
            elif self.spike_bf16:  # STEM_SPIKE=bf16: as below, the spikes in bf16
                y = lif4_bf16_op(*ys, ys[0].new_full((1, 1, self.k * ys[0].shape[2], c_), self.lif_fill,
                                                     dtype=torch.bfloat16))
            elif self.k == 2:  # STEM_LIF4 with two convs: two time steps per output (make_lif2)
                y = lif2_op(*ys, ys[0].new_full((1, 1, self.k * ys[0].shape[2], c_), self.lif_fill))
            else:  # STEM_LIF4: the LIF reads the four outputs, one per time step, and writes the spikes
                y = lif4_op(*ys, ys[0].new_full((1, 1, self.k * ys[0].shape[2], c_), self.lif_fill))
            return y.reshape(t_ * n_, h_, w_, c_).permute(0, 3, 1, 2).reshape(t_, n_, c_, h_, w_)

    m.conv1 = SplitStem(m.conv1, m.bn1, STEM_SPLIT, tuple(xin.shape))
    m.bn1 = torch.nn.Identity()
    if os.environ.get("STEM_LIF4") == "1":
        # one conv per time step (k = T = 4): the stem LIF reads the four conv outputs directly (make_lif4), so
        # they need no concat (0.5 ms, S24); the LIF moves into the stem module and sn1 becomes identity. With k = 2
        # (two time steps per conv, 0.1 ms faster convs, S32) it reads the two outputs (make_lif2)
        assert STEM_SPLIT in (2, 4) and T == 4 and os.environ.get("NEURON") == "ttl", \
            "STEM_LIF4 needs STEM_SPLIT=4 or 2, T=4, NEURON=ttl"
        m.conv1.lif_fill = m.sn1.fill
        m.sn1 = torch.nn.Identity()
    if os.environ.get("STEM_SPIKE") == "bf16":
        # bf16 spikes go straight into the maxpool, which tt-mlir runs in bf16 (it typecast the fp32 spikes before,
        # 0.35 ms, S24); its output goes back to fp32 for layer1, as it did before (a to_layout after the maxpool)
        assert m.conv1.lif_fill is not None and STEM_SPLIT == 4, "STEM_SPIKE=bf16 needs STEM_LIF4=1, STEM_SPLIT=4"
        m.conv1.spike_bf16 = True

        class FloatOut(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x).float()

        m.maxpool = FloatOut(m.maxpool)
    print(f"stem conv split into {STEM_SPLIT} convs over the T*N images, its BN folded in"
          + (f", its LIF reading the {STEM_SPLIT} outputs" if m.conv1.lif_fill is not None else "")
          + (" and writing bf16 spikes" if m.conv1.spike_bf16 else ""), flush=True)
if os.environ.get("AVGPOOL") == "matmul":
    # The final average pool as a matmul with a constant 0/1 matrix over the rows of the last LIF's output, then
    # one multiply by 1/(H*W) (S34). tt-mlir lowered AdaptiveAvgPool2d((1, 1)) to a reshape of the [1, T, N*H*W, C]
    # LIF output to [T*N, 1, H*W, C] (49 rows per image do not fill tiles, so the data moves), a mean and another
    # reshape (0.11 ms in all); the matmul reads the LIF output where it is. The sums of the spike counts (integers)
    # are exact.
    class MatmulAvgPool(torch.nn.Module):
        step_mode = "m"

        def __init__(self, n, hw):
            super().__init__()
            self.register_buffer("p", torch.kron(torch.eye(n), torch.ones(1, hw)))  # [N, N*H*W]

        def forward(self, x):
            t_, n_, c_, h_, w_ = x.shape
            xs = x.reshape(t_ * n_, c_, h_, w_).permute(0, 2, 3, 1).reshape(t_, n_ * h_ * w_, c_)
            return (torch.matmul(self.p, xs) * (1.0 / (h_ * w_))).reshape(t_, n_, c_, 1, 1)

    m.avgpool = MatmulAvgPool(N, (H // 32) ** 2)
    print("final average pool as a matmul", flush=True)
if PREFIX:
    m = Prefix(m, PREFIX)
    print(f"running the model up to {PREFIX}", flush=True)
if os.environ.get("INPUT_LAYOUT") == "nhwc":
    # hand the device the input as [1, 1, T*N*H*W, C], the layout the stem conv reads, and permute it back to NCHW
    # as the model's first op, so that tt-mlir folds that permute with the one it puts before the conv (S23); without
    # this the device tiles the [T, N, C, H, W] input and permutes it to NHWC in every step (about 7 ms, S22)
    class InputNHWC(torch.nn.Module):
        def __init__(self, inner, shape):
            super().__init__()
            self.inner, self.shape = inner, shape

        def forward(self, *xs):
            if STEM_SPLIT > 1:  # one [1, 1, b*H*W, C] per group: the split stem takes them as they are
                return self.inner(xs)
            t_, n_, c_, h_, w_ = self.shape
            return self.inner(xs[0].reshape(t_ * n_, h_, w_, c_).permute(0, 3, 1, 2).reshape(self.shape))

    m = InputNHWC(m, tuple(xin.shape))
    xin = xin.permute(0, 1, 3, 4, 2).reshape(STEM_SPLIT, 1, 1, -1, xin.shape[2])
    xin = tuple(g.contiguous() for g in xin)
    print(f"input handed over as {len(xin)} x {list(xin[0].shape)} (NHWC)", flush=True)
m = m.to(dev)
if mode == "compile":
    torch._dynamo.reset()
    m = torch.compile(m, backend="tt")
xd = tuple(g.to(dev) for g in xin) if isinstance(xin, tuple) else (xin.to(dev),)
torch_xla.sync(wait=True)


def step():
    out = m(*xd)
    functional.reset_net(m)
    return out


samples, first, split, all_steps = [], None, [], []
with torch.no_grad():
    pause = (os.environ.get("PAUSE", "-1:0") + ":sleep").split(":")
    pause_at, pause_s, pause_kind = int(pause[0]), float(pause[1]), pause[2]
    for i in range(steps):
        if i == pause_at:
            if pause_kind == "spin":
                end = time.perf_counter() + pause_s
                while time.perf_counter() < end:
                    pass
            else:
                time.sleep(pause_s)
        ep = time.time()  # epoch of the step start, to line steps up with tools/tt-telem-poll.py
        ru0 = resource.getrusage(resource.RUSAGE_SELF)  # all threads of the process
        t0 = time.perf_counter()
        out = m(*xd)
        t1 = time.perf_counter()
        functional.reset_net(m)
        t2 = time.perf_counter()
        torch_xla.sync(wait=True)
        if os.environ.get("WAIT") == "cpu":
            out.cpu()
        t3 = time.perf_counter()
        dt = (t3 - t0) * 1e3
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20  # host RSS, MiB
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime - ru0.ru_utime + ru1.ru_stime - ru0.ru_stime) * 1e3
        sys_ms = (ru1.ru_stime - ru0.ru_stime) * 1e3
        vcsw, ivcsw, minflt = ru1.ru_nvcsw - ru0.ru_nvcsw, ru1.ru_nivcsw - ru0.ru_nivcsw, ru1.ru_minflt - ru0.ru_minflt
        all_steps.append([round(ep, 3), round(dt, 2), round(rss), round(cpu, 1), round(sys_ms, 1), vcsw, ivcsw, minflt])
        if i >= (2 if TRACE else 1):
            split.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
        if i == 0:
            first = dt
            out0 = out.cpu()
        elif i >= (2 if TRACE else 1):
            samples.append(dt)
        print(f"step {i} at {ep:.3f}: {dt:.1f} ms (call {(t1 - t0) * 1e3:.1f}, reset {(t2 - t1) * 1e3:.1f}, sync {(t3 - t2) * 1e3:.1f}, rss {rss:.0f} MiB, cpu {cpu:.1f} ms (sys {sys_ms:.1f}), csw {vcsw}/{ivcsw}, minflt {minflt})", flush=True)
    # pipelined: enqueue all steps, sync once (closer to tt-xla's vision benchmark)
    k = max(steps - 1, 1)
    t0 = time.perf_counter()
    outs = [step() for _ in range(k)]
    t1 = time.perf_counter()
    torch_xla.sync(wait=True)
    [o.cpu() for o in outs[-1:]]
    t2 = time.perf_counter()
    pipelined = (t2 - t0) * 1e3 / k
    # submit: the host returns from the k calls (it blocks while the previous execution is still being submitted);
    # drain: how long the device then takes to finish, i.e. how far it is behind the host (S23)
    pipe_split = {"submit_ms_per_step": (t1 - t0) * 1e3 / k, "drain_ms": (t2 - t1) * 1e3}

pcc = torch.corrcoef(torch.stack([out0.flatten(), ref.flatten()]))[0, 1].item()
if os.environ.get("SAVE_LOGITS"):
    torch.save({"device": out0, "cpu": ref}, os.environ["SAVE_LOGITS"])
counters = {}
if mode == "compile":
    from torch._dynamo.utils import counters as dc
    counters = {k: dict(v) for k, v in dc.items() if k in ("stats", "graph_break")}
res = {
    "mode": mode, "neuron": os.environ.get("NEURON", "simple"), "lif_layout": LIF_LAYOUT, "lif_out": os.environ.get("LIF_OUT", "distinct"), "bn": os.environ.get("BN", "train"), "stem_pad": STEM_PAD, "stem_fold": os.environ.get("STEM_FOLD") == "1", "stem_split": STEM_SPLIT, "stem_lif4": os.environ.get("STEM_LIF4") == "1", "stem_spike": os.environ.get("STEM_SPIKE", "fp32"), "lif_grid": os.environ.get("LIF_GRID", "auto"), "lif_res": os.environ.get("LIF_RES") == "1", "input_layout": os.environ.get("INPUT_LAYOUT", "nchw"), "avgpool": os.environ.get("AVGPOOL", "adaptive"), "wait": os.environ.get("WAIT", "sync"), "optimization_level": OPT, "trace": TRACE, "batch": N, "T": T, "image_size": H, "steps": steps,
    "first_ms": first, "median_ms": statistics.median(samples) if samples else None,
    "min_ms": min(samples) if samples else None, "max_ms": max(samples) if samples else None,
    "pipelined_ms_per_step": pipelined, "pipelined_split": pipe_split,
    "split_median_ms": {k: statistics.median(v[j] for v in split) for j, k in enumerate(("call", "reset", "sync"))} if split else None,
    "logits_pcc_vs_cpu": pcc, "logits_maxabs_vs_cpu": (out0 - ref).abs().max().item(),
    "prefix": PREFIX, "mismatch_vs_cpu": int((out0 != ref).sum()), "elements": ref.numel(),
    "rel_err_vs_cpu": ((out0 - ref).norm() / ref.norm()).item(),
    "dynamo": counters,
    "steps_epoch_ms": all_steps,
}
print("RESULT " + json.dumps(res), flush=True)
