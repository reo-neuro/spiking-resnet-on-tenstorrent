# Spiking ResNet on Tenstorrent

SEW-ResNet18 (SpikingJelly 2.0) inference on a Tenstorrent Blackhole p150a, compiled with tt-xla (torch_xla + the
`pjrt-plugin-tt` PJRT plugin), with the multi-step LIF neuron replaced by a fused tt-lang kernel. Over the course of the
work the time per inference step went from 222.8 ms to 9.44 ms; an NVIDIA A30 with TF32 takes 7.52 ms for the same
step. The repository holds the kernel, the scripts used to run and measure the model on both devices, and the notes on
what the time was spent on and which constraints of the toolchain we ran into.

## Results

One step = one inference of T = 4 time steps (`reset_net` to logits), SEW-ResNet18 (ADD connections), 224 x 224
input, batch 8, fp32 tensors, BatchNorm in inference mode. "Pipelined" submits k steps back to back and waits once
(time per step); "synced" reads the logits back to the host after every step.

| Stage | Pipelined | Synced |
| --- | --- | --- |
| Original `SimpleLIFNode` (LIF as small TTNN ops) | 222.8 ms | 255.8 ms |
| LIF as one tt-lang kernel, called on the BN output layout (NCHW) | 70.1 ms | 85.7 ms |
| LIF kernel called on the conv output layout | 31.5 ms | 43.0 ms |
| Remove the reshapes around the LIF, layer4 passed as `[1, T, R, C]` | 28.3 ms | 34.6 ms |
| Trace | 21.1 ms | 27.2 ms |
| Input laid out for the conv on the host (under trace) | 17.1 ms | 22.7 ms |
| Stem conv folded 2 x 2 | 14.5 ms | 19.6 ms |
| Stem conv split per time step, read directly by the LIF | 11.9 ms | 17.3 ms |
| Leaner LIF kernel on 110 cores | 10.1 ms | 15.5 ms |
| Residual add fused into the LIF | 9.59 ms | 14.2 ms |
| Stem conv as 16 images x 2, average pool as a matmul | **9.44 ms** | 15.0 ms |
| (reference) A30, TF32 and `cudnn.benchmark` | 7.52 ms | 7.63 ms |
| (reference) A30, strict fp32 as in the SpikingJelly benchmark | 14.75 ms | — |

Scope and caveats:

- Speed only. The weights are the randomly initialized ones and the inputs are random, as in SpikingJelly's
  benchmark; accuracy with trained weights was not evaluated.
- Correctness was checked against the CPU in the same process, layer by layer. Changes that should not alter the math
  (removing reshapes, splitting, fusing) were required to keep the logits bit-identical. The conv outputs on the p150a
  carry about 10 mantissa bits, which flips spikes near the threshold. When the LIF kernel was introduced, the logits
  correlated with the CPU at 0.911 (0.908 for the original `SimpleLIFNode` on the device); CPU-only runs whose layer
  outputs are perturbed by rounding-sized noise level off at about 0.92.
- Host-side preprocessing used by the later stages (input reordering, channel padding, folding, splitting per time
  step; about 6.6 ms under trace) is not included in the p150a times.
- The A30 reference is the same model built the same way, with SpikingJelly's Triton `LIFNode`. The SpikingJelly
  benchmark itself disables TF32; both values are shown.
- Versions: `pjrt-plugin-tt` 1.4.0 (with its bundled tt-lang 1.1.5), torch 2.11.0, torch-xla 2.9.0, p150a firmware
  19.15.0. Several of the constraints below are specific to these versions.

## What we found

- Before trace, the step time was set by the host submitting operations one by one (roughly 70–360 µs per operation,
  12–72 times the A30's). Writing a neuron as element-wise ops costs about 7 ops per layer per time step, which made
  the LIF the largest source of operations. Fusing each LIF layer into one kernel is the first thing to do.
- After trace, replaying the TTNN IR op by op predicts the step time well (9.19 ms busy vs 9.44 ms measured), so the
  replay breakdown can choose the next change.
- On the device, reorderings (`reshape`, `permute`, `to_layout`) of shapes that are not multiples of the 32 x 32 tile
  cost more than the convs at first (about 1 ms each). Calling the neuron kernel on the layout of the ops around it
  removes them.
- Spikes are 0/1 and the SEW sums are small integers, so they are exact in any precision. Fusing the residual add into
  the LIF and computing the average pool as a matmul with a 0/1 matrix left the logits bit-identical.
- What remains is conv and LIF time against DRAM bandwidth. Keeping activations in L1 would cut about 1.9 ms more
  (measured by replaying the IR with L1 placement), but tt-xla 1.4.0 offers no way to request it.

The full notes (in Japanese) are in `docs/ja/`: `speedup-guide.md` (how to approach another SNN, what worked and what
did not), `discrepancies.md` (D1–D32: assumptions that turned out wrong, toolchain constraints and bugs) and
`setup.md` (environments, script reference, the LIF kernel). S-numbers in the notes and code comments refer to entries
of our lab notebook, which is not published.

## Repository layout

| Path | Contents |
| --- | --- |
| `tools/ttl_lif.py` | The multi-step LIF kernels in tt-lang (`make_lif`, `make_lif4`, `make_lif2`, `make_lif_add`, `make_lif_nchw`) |
| `tools/xla-sew18.py` | Runs and times SEW-ResNet18 on the p150a through tt-xla, with switches for every change in the table |
| `tools/a30-sew18.py`, `tools/a30-sew18-profile.py` | The same model on the A30, and its per-layer GPU time |
| `tools/ttnn-replay-ir.py`, `tools/ttnn-replay-chain.py` | Replay the TTNN IR written by tt-xla op by op (or chained, with L1 placement) under the device profiler |
| `tools/breakdown-sew18.py`, `tools/chain-csv.py` | Per-category and per-layer breakdowns and lower bounds |
| `tools/ttl-lif-bench.py` | Device time of the LIF kernels on the SEW-ResNet18 shapes |
| `tools/xla-op-hostcost.py`, `tools/thread-sampler.py` | Host cost per operation; per-thread CPU frequency while waiting |
| `tools/lif-d22-emul.py`, `tools/xla-conv-check.py`, `tools/xla-stem-pad.py`, `tools/xla-bn-eval.py`, `tools/xla-op-backward.py`, `tools/sj_xla.py` | Reproducers and workarounds for the constraints in `discrepancies.md` |
| `docs/ja/` | Notes (Japanese) |

## Reproducing the best configuration

Environments are described in `docs/ja/setup.md`. Paths are written as environment variables: `$VENV_XLA140` (Python
3.12 venv with `pjrt-plugin-tt==1.4.0`, torch-xla 2.9.0 and SpikingJelly 2.0.0 installed with `--no-deps -e`),
`$UV_PY312_LIB` (the `lib` directory of the uv-managed Python 3.12, needed by torch_xla to find libpython), `$VENV_A30`
(torch 2.11.0+cu128 with SpikingJelly 2.0.0) and `$DATA` (output directory).

```bash
# p150a: 20 pipelined steps, batch 8, T = 4, 224 x 224, optimization level 1, trace
NEURON=ttl LIF_LAYOUT=nhwc4 BN=eval STEM_PAD=32 STEM_FOLD=1 STEM_SPLIT=2 STEM_LIF4=1 INPUT_LAYOUT=nhwc \
LIF_GRID=11,10 LIF_RES=1 AVGPOOL=matmul WAIT=cpu LD_LIBRARY_PATH=$UV_PY312_LIB \
  $VENV_XLA140/bin/python -I tools/xla-sew18.py compile 8 4 20 224 1 trace

# A30 reference (TF32, cudnn.benchmark)
CUDA_VISIBLE_DEVICES=0 TF32=1 CUDNN_BENCH=1 BN=eval $VENV_A30/bin/python -I tools/a30-sew18.py LIFNode 8 4 224 30
```

We pinned the host threads to the NUMA node of each card and took the median of three runs.

## Next

We are moving to SRAM-resident, time-stepped SNN execution on Blackhole: weights, membrane potentials and spikes stay
in the Tensix cores' L1 across time steps, with the cores synchronized once per step (in progress).

## License

Apache-2.0 (see `LICENSE`). The scripts use SpikingJelly, tt-xla, tt-lang and TT-NN, which are also Apache-2.0 and are
not included here.

## Citation

See `CITATION.cff`.
