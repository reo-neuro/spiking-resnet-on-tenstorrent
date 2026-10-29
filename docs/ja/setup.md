# 環境と実行の手順

記録を取った環境は、Supermicro SYS-741GE-TNRT（Ubuntu 22.04、Xeon Max 9468 × 2）に Tenstorrent Blackhole p150a（ファームウェア 19.15.0）と NVIDIA A30 を挿したもの。パスは環境変数で書く。`$SNN` は SpikingJelly の clone と venv を置く場所、`$VENV_A30`、`$VENV_XLA`、`$VENV_XLA140` は下の 3 つの venv、`$UV_PY312_LIB` は uv の Python 3.12 の `lib`、`$DATA` は計測の結果を置く場所。経緯の番号のうち、D は `discrepancies.md` の、S は公開していない作業記録の番号である。

## A30 の環境

公式のインストールガイド（clone の `docs/source/tutorials/en/install.rst`）で検証済みとされている組み合わせに合わせた。

- Python 3.11.16（uv）、torch 2.11.0+cu128、torchvision 0.26.0、Triton 3.6.0
- SpikingJelly は clone から editable で入れた（`uv pip install --editable ".[triton]"`）。ネイティブ CUDA の拡張はまだビルドしていない

```bash
cd $SNN
uv venv --python 3.11 venv
VIRTUAL_ENV=$PWD/venv uv pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
cd spikingjelly && VIRTUAL_ENV=../venv uv pip install --editable ".[triton]" pytest
```

ニューロンのバックエンドはテンソルのデバイスから自動で選ばれる。今の A30 では LIF に Triton が選ばれる。確かめ方は次のとおり。

```bash
../venv/bin/python -c 'import torch; from spikingjelly.activation_based import functional; print(functional.neuron_implementation("lif", torch.device("cuda:0")))'
```

## A30 での計測

`benchmark/benchmark_snn_single_gpu.py` を使う。入力が乱数なので、速さだけを比べるためのもので、精度は見られない。clone の直下で実行する。

```bash
../venv/bin/python benchmark/benchmark_snn_single_gpu.py case --model sew_resnet18 \
  --phase inference --execution eager --batch-size 32 --warmup 10 --steps 30 \
  --require-gpu-name A30 --output $DATA/a30_sew18_inference_eager_bs32_T4.json
```

モデルは `spiking_vgg16_bn`、`sew_resnet18`、`spikformer_ti`、`spikformer_s` から選べる。`--execution` は `eager`、`compile`、`cuda_graph` から選ぶ。

2026-10-27 の結果（SEW-ResNet18、LIF、fp32、224×224、T=4、batch 32、Triton、eager）。

| phase | 中央値 | p10〜p90 | ピークのメモリ |
| --- | --- | --- | --- |
| 推論 | 51.1 ms | 51.0〜51.4 ms | 1.57 GB |
| 学習（SGD） | 169.5 ms | 168.8〜170.0 ms | 6.25 GB |

ピークのメモリは `peak_allocated_bytes`（10^9 B で換算）。

ベンチマークは TF32 を切る（`benchmark_snn_single_gpu.py` の 502〜503 行目）ので、conv は厳密な fp32 で計算する。p150a の conv の出力は仮数 10 ビット相当なので、精度をそろえて比べるときは、TF32 を許した時間を `tools/a30-sew18.py` で測る（`discrepancies.md` の D26）。2026-10-28 の結果（SEW-ResNet18、`LIFNode`、推論モードの BN、224×224、T=4、batch 8、eager。S20）。

| 条件 | 1 ステップ |
| --- | --- |
| ベンチマークのまま（厳密な fp32） | 14.75 ms（cuda_graph 14.58 ms、compile 19.76 ms） |
| TF32 を許し、`cudnn.benchmark=True` | 7.5 ms |

TF32 を許すと、ウォームアップの後も 12〜20 ステップは約 10 ms と遅いので、それより長く回してから測る。p150a との比較の基準は TF32 の値とする（ユーザーの決定、2026-10-28。ベンチマークのままの値は併記する）。

## p150a（torch_xla）の環境

- venv: `$VENV_XLA`（Python 3.12.14、`pjrt-plugin-tt==1.5.0.dev20260831000501`、torch 2.11.0+cpu、torch-xla 2.9.0。SpikingJelly は `--no-deps -e`）
- venv（1.4.0）: `$VENV_XLA140`（2026-10-27。`venv-xla` と同じ構成で、`pjrt-plugin-tt==1.4.0`、torch-xla 2.9.0+git56f8e61。1.5.0.dev20260831 では level 1・2 の Conv が通らないため。S8・S10）
- 実行時に `LD_LIBRARY_PATH=$UV_PY312_LIB` を付ける（torch_xla が libpython を探せないため）

```bash
LD_LIBRARY_PATH=$UV_PY312_LIB \
  $VENV_XLA/bin/python tools/xla-cnn-step2.py 5 lif manual
```

p150a で使うときの制約（2026-10-27、S2）は次のとおり。

- `LIFNode` などの登録済みのニューロンは、CPU と CUDA の実装しかないので xla では動かない。純粋な PyTorch の `SimpleLIFNode` を使う
- Conv の直後の `nn.BatchNorm2d`（学習モード）は backward が壊れる。`tools/sj_xla.py` の `ManualBN2d` を使う。eval モードの BN は forward ならそのまま動く（optimization_level 0）
- SEW-ResNet18 は batch 12 以上で実行時に止まる（`Invalid subtile broadcast type`。1.5.0 は 16 で確認、1.4.0 は 12 と 16 で確認）
- optimization_level 1・2 は、Conv を含むとコンパイルできない（1 は `ttnn.permute failed validation`、2 はコンパイラの segfault）。Conv 1 個でも、bf16 でも、公式の例 `examples/pytorch/mnist_performant.py` でも同じ（S6）。1.5.0.dev20260831 の退行で、1.4.0（`venv-xla140`）なら level 1 が通る（S8、S10）。1.4.0 でも level 2 はコンパイルが終わらない
- trace（`enable_trace`）は、小さいモデル（Conv＋BN＋ReLU）では level 0〜2 のどれでもコンパイルで abort する（`Memory layout is required for non-SystemMemory buffer type`。Linear だけなら動く。S6、S8）。ただし SEW-ResNet18（level 1、推論モードの BN）は trace で動き、ホストの送り出しの費用（trace なしで 1 ステップ約 22〜27 ms）がほぼ消える（`xla-sew18.py` の `trace` の引数。S23、D15）
- optimization_level 1 は、入力が 3〜16 チャネルの stem の conv（SEW-ResNet18 の 7×7、stride 2）を誤って計算する（CPU との相関 0.28。32 チャネルにゼロで埋めると正しい。ほかの conv は正しい）。`xla-sew18.py` は `STEM_PAD=32` で回避する（`discrepancies.md` の D24）
- SpikingJelly のベンチマークの推論は、モデルを推論モードにして流す（`model.train(False)`）。p150a でも BN は推論モードで測る（`xla-sew18.py` の `BN=eval`。D23）。推論モードの BN は、tt-mlir が conv に畳み込む
- 時間の測り方: `torch_xla.sync(wait=True)` はデバイスの完了を待たない。1 ステップの時間は、出力を `.cpu()` で読み戻すまでを測る（`xla-sew18.py` と `xla-ttl-lif.py` の `WAIT=cpu`。`discrepancies.md` の D21）
- 1 ステップごとに同期する時間は、ホストの CPU の省電力で 3〜5 ms 延び、回ごとにぶれる。待つたびにスレッドが眠り、CPU が 800 MHz（`acpi-cpufreq` と `schedutil` の最低）のまま動くため。SMT の兄弟（156〜167）に SCHED_IDLE の空回り（`taskset -c <cpu> chrt -i 0 python3 -c 'while 1: pass'`）を置き、計測を 60〜71 に固定すると消える（SEW-ResNet18 で 14.2 → 11.0 ms）。まとめて同期にはほとんど効かない（9.59 → 9.42 ms）。既定の値は省電力のまま測る（S31、D31）
- tt-xla 1.4.0 の trace の入力は、常にホストの行優先として扱われ、trace の中でタイルに直される。デバイスのタイルのテンソルを渡しても同じ（S33、D32）

| スクリプト | 内容 |
| --- | --- |
| `tools/xla-op-backward.py` | 演算子ごと（とその組み合わせ）の backward を CPU と比べる。`ONLY=` で選び、`OPTS=` で tt-xla のコンパイルオプションを渡す |
| `tools/xla-bn-eval.py [lazy\|compile] [level]` | Conv → eval の BN の forward を、lazy tensor と `torch.compile(backend="tt")` で CPU と比べる |
| `tools/xla-sew18.py [compile\|lazy] [batch] [T] [steps] [size] [level] [trace\|notrace] [perf_metrics_file]` | SEW-ResNet18 の推論を SpikingJelly のベンチマークと同じ組み方（ニューロンだけ `SimpleLIFNode`）で測り、logits を CPU と比べる。`NEURON=ttl` で LIF を `ttl_lif.py` のカーネルに差し替える（`venv-xla140`）。並べ方は `LIF_LAYOUT` で選ぶ。推論モードの BN では `nhwc4`（conv の並びのまま `make_lif` に渡す）、学習モードの BN では `nchw`（BN の出力の形のまま `make_lif_nchw` に渡す）。既定の `flat`（`[T·R, Cc]` に並べ替える）と `nhwc` は不利。`NEURON=none` で恒等写像（CPU の基準も恒等写像）、`NEURON=reshape` で `flat` と同じ並べ替えの間に `x * 0.5` を置く。`BN=eval` で推論モード（ベンチマークと同じ）、`BN=none` で BN を外す。`STEM_PAD=32` で stem の conv を埋める（D24）。`STEM_FOLD=1` で、入力をホストで 2×2 ずつチャネルに折りたたみ、stem を 4×4、stride 1 の conv にする（`STEM_PAD=32` と組で使う。16 チャネルでは D24 が起きる）。`INPUT_LAYOUT=nhwc` で、入力をホストで `[1, 1, T·N·H·W, C]` に並べてからデバイスに置く（S23）。`STEM_SPLIT=k`（`BN=eval` と組で使う）で、stem の BN をホストで conv に畳み、T·N 枚を k 組に分けて k 個の conv にする（S24。32 枚のままだと L1 に収まらず、DRAM を通る分割になる）。`INPUT_LAYOUT=nhwc` と一緒に使うと、入力は組ごとの k 個の `[1, 1, b·H·W, C]` になる。`STEM_LIF4=1`（`STEM_SPLIT=4` か `2`、T = 4、`NEURON=ttl` と組で使う）で、stem の LIF が 4 つ（`make_lif4`）か 2 つ（`make_lif2`。S32）の conv の出力を直接読む（結合が要らない）。今の最良は `STEM_SPLIT=2`（16 枚の conv 2 つ。8 枚 × 4 より 0.1 ms 速い）。`LIF_RES=1`（`LIF_LAYOUT=nhwc4` と組）で、SEW のブロックの残差の足し算を 2 つ目の LIF のカーネルに取り込む（`make_lif_add`。S30）。`AVGPOOL=matmul` で、最後の平均プーリングを 0/1 の行列との積にする（並べ替えが要らない。S34）。今の最良の組み合わせは README の「Reproducing the best configuration」にある。`LIF_GRID=11,10` で `make_lif` と `make_lif4` を p150a の 110 コアで動かす（指定しないと、tt-xla の橋渡しの仮の device に合わせて 64 コアになる。D28）。`STEM_SPIKE=bf16`（`STEM_LIF4=1` と組）で stem の LIF が bf16 のスパイクを書く（S26。tt-xla の経路ではデバイスが止まった。D29。使わない）。`WAIT=cpu` で毎ステップ logits を読み戻す（時間を測るときは必ず付ける）。`PREFIX=<段>`（`conv1`、`bn1`、`sn1`、`maxpool`、`layer1`〜`layer4`）でその段までだけを流し、出力を CPU と要素ごとに比べる。`LIF_OUT=zeros` で、LIF の出力先を共有してしまう元の作り方に戻す（D25 の再現用）。`SAVE_LOGITS=<file.pt>` で 1 ステップ目の logits と CPU の基準を保存する。`PAUSE=<step>:<秒>[:spin]` で途中で休ませ、`EXPORT=<dir>` で IR を書き出す。各ステップの時刻、CPU 時間、RSS、まとめて同期の送り出しと追いつき待ち（`pipelined_split`）も出す |
| `tools/lif-d22-emul.py [trunc\|rne] [batch] [T] [size] [saved_logits.pt ...]` | CPU だけで、tt-xla の経路の tt-lang の LIF（CB を読むたびに仮数 10 ビットへ切り捨てる。D22）をまねて SEW-ResNet18 を流し、層ごとのスパイクの反転と logits を正確な LIF と比べる。`xla-sew18.py` の `SAVE_LOGITS` で保存したデバイスの logits とも比べる |
| `tools/xla-conv-check.py <level> <module path> [spikes\|randn]` | SEW-ResNet18 の conv を 1 つだけ、デバイスと CPU で比べる（D24 の確認） |
| `tools/xla-stem-pad.py <level> <cin> [batch]` | stem の conv の入力チャネルを cin にゼロで埋めて、デバイスと CPU で比べる（D24 が直ったかの確認） |
| `tools/xla-op-hostcost.py <level> <add\|lif\|lif_alg\|conv\|bn\|conv_bn_lif> <K...>` | 同じ演算を K 個つないだ小さいグラフで、演算 1 個あたりのホストの時間を測る。`lif_alg` は演算を減らした LIF の式。`OPTS=` でコンパイルオプションを足し、`PERF=` で演算の数を書き出す。`SHAPE=N,C,H,W` で実際の大きさにし、`WAIT=cpu` で出力を読み戻してデバイスの完了まで測る（まとめて同期と、その送り出しと追いつき待ちも出す）。`REDUCE=1` で出力を足して読み戻しを小さくする（S23） |
| `tools/ttl_lif.py [T] [R] [C] [fp32] [split4 [bf16out] \| split2 \| res]` | LIF（`SimpleLIFNode` と同じ式）を T ステップまとめて行う tt-lang のカーネル。`make_lif` は `[1, 1, T·R, C]`（conv の出力の並び）と `[1, T, R, C]`（R が 32 の倍数でないとき）、`make_lif4` は T = 4 の各ステップを別々の `[1, 1, R, C]` で（`split4` で単体の試験。`bf16_out=True`（`bf16out`）でスパイクを bf16 で書く。S26）、`make_lif2` は 2 時刻ずつの `[1, 1, 2·R, C]` 2 つで（`split2`。S32）、`make_lif_nchw` は `[T·N, C, H, W]`（BN の出力の並び）を受ける。`make_lif_add` は `make_lif` に残差 r を足して `s + r` を書く（`res`。S30）。単体では ttnn で実機に流して CPU と比べる（コンテナで `run-test.sh --hw`。閾値ちょうどの例も見る）。fp32 だけ。tt-lang 1.1.5 と 1.1.10 の両方で通る書き方にしてある（`discrepancies.md` の D19） |
| `tools/ttl-lif-bench.py <kernel.py>:<factory>:<flat\|nchw\|nhwc\|nhwcres\|split4\|split4bf16\|split2>[,...] [sew18\|B:C:H:W,...] [reps]` | コンテナで、LIF のカーネルのデバイス時間を、デバイスのプロファイラのカーネルの区間で形ごとに測る。複数のバージョンを同じ入力で流し、出力がビット単位で同じかも見る。`split4` は `make_lif4` を 4 入力で測る（`factory()` を T なしで呼ぶ）。`split4bf16` は bf16 の出力で測る（`factory(bf16_out=True)`）。`split2` は `make_lif2` を 2 入力で、`nhwcres` は `make_lif_add` を残差（0〜3 の整数）つきで測る（S30、S32）。プロファイラは CB の操作ごとに計測を足すので、小さなブロックを多く回す形では時間が過大に出る（S16）。`LIF_MEM=l1`（L1 の interleaved）、`l1h`（11×10 の格子に行で分けた L1）で、入出力を L1 に置いて測る（S29） |
| `tools/ttnn-replay-ir.py <ttnn.mlir> <out.json> [reps] [T]` | コンテナで、tt-xla が書き出した TTNN の IR の主関数の演算を 1 つずつ ttnn で流し、演算ごとのデバイス時間（稼働と区間）をプロファイラで測る（tt-xla 1.4.0 はプロファイラが使えないため。コンテナの tt-metal はバージョンが違うので目安）。trace ありで書き出した IR は、trace 用の関数（`@trace_0_main`）を流す。IR はホームの下に写して渡す（コンテナから `/work` と `/tmp` は見えない。`~/bench-data` も `/work` へのリンクなので見えない。パスは `$HOME` で絶対パスにする）。`ttnn.generic` は入力の数と形で `make_lif`、`make_lif4`、`make_lif_add`、`make_lif2` を選ぶ（S30、S32）。`[1, 4, 392, 512] → [4, 392, 512]` の `reshape`（`AVGPOOL=matmul`）はコンテナの ttnn では流せない（S34） |
| `tools/ttnn-replay-chain.py <ttnn.mlir> <out.json> <policy> [reps] [T]` | コンテナで、tt-xla が trace ありで書き出した IR の trace の関数を、演算の出力を次の演算へつないで流し（IR の `deallocate` も守る）、演算ごとのデバイス時間を測る。出力の置き場所を方針で選ぶ（`mem=dram\|l1\|l1s`、`l1_from`、`dbuf`、`lif_tiles`、`dram_ops`。スクリプトの冒頭に一覧）。L1 に入らない演算は、出力を L1 の interleaved、DRAM の順に落とす。それでも動かなければ、L1 にあるテンソルをすべて DRAM へ移してから、出力を DRAM に置いて流す。最初の 1 回で LIF のスパイクを CPU と照らし合わせる（S29） |
| `tools/chain-csv.py <out dir> <key>::<label>::<chain.json> ...` | `ttnn-replay-chain.py` の結果を、時間の内訳のダッシュボード（含めていない）が読む `l1policies.csv`（置き方ごとの 1 ステップの稼働、L1 から移した演算の数）と `l1chain.csv`（置き方ごとの種類ごとの稼働。種類の分け方は `breakdown-sew18.py` と同じ）にする（S29） |
| `tools/a30-sew18-profile.py [LIFNode\|SimpleLIFNode] [batch] [T] [size] [steps] [out.json]` | A30 で、層ごとの GPU の時間を torch.profiler で測る（`a30-sew18.py` と同じ組み方とスイッチ） |
| `tools/breakdown-sew18.py <replay.json> <a30 tf32.json> <a30 fp32.json> <out.json> [<比べる前の replay.json> <csv dir>]` | `ttnn-replay-ir.py` と `a30-sew18-profile.py` の結果を、種類ごと、層ごとの内訳にまとめ、時間の内訳のダッシュボード（含めていない）が読む CSV 8 つを書く。比べる前のバージョンは「p150a（始める前）」の行になる。測った 1 ステップの時間と高速化の経過は、スクリプトの中の `STEPS` と `HISTORY`。1 ステップの下限の見積もり（`bounds.csv` と前提の値の `bound_params.csv`。S28）も出す。前提（DRAM の帯域、演算性能）はスクリプトの中の定数 |
| `tools/a30-sew18.py [LIFNode\|SimpleLIFNode] [batch] [T] [size] [steps]` | A30 で SEW-ResNet18 の推論を `xla-sew18.py` と同じ組み方で測る。`TF32`、`CUDNN_BENCH`、`BN`、`GRAD` で、ベンチマークと `a30-hostcost.py` で違う条件を 1 つずつ切り替え、CUDA event、1 ステップごとに同期、まとめて同期の 3 つの計時を出す |
| `tools/sj_xla.py` | p150a 用の回避策（`ManualBN2d`） |
| `tools/thread-sampler.py <pid> [interval] [duration]` | 動いているプロセスのスレッドごとの CPU 使用、載っている CPU、その周波数を記録する（読み取りのみ） |

## LIF の融合カーネル（tt-lang、2026-10-27）

`tools/ttl_lif.py` の `make_lif(T)` は、SpikingJelly の `SimpleLIFNode`（tau=2、`decay_input=True`、`v_threshold=1`、`v_reset=0` の hard reset、`detach_reset=True`）の T ステップを、1 つの tt-lang のカーネルで計算する。推論だけで、fp32 を前提にしている（S14）。

- 入出力: `make_lif_nchw(T)` は `[T·N, C, H, W]`（BN の出力の形のまま。H と W はタイルに余白を足して置かれている）を受ける。時刻 t の画像 n は `t·N + n` 番目なので、各 (n, c) の面は別々のタイルで、時刻の境目は必ずタイルの境目に来る。1 時刻ぶんの N·C 枚の面を全コアに等しく分け、各コアが自分の面について T ステップを順に処理し、膜電位は DFB（L1）に置いたまま持ち越す。余白の要素も計算する（読み出されない）
- 計算（`make_lif_nchw`）: h₀ = `x * 0.5`、hₜ = `v + (x * 0.5 − v * 0.5)`（`SimpleLIFNode` の `v + (x − v) / 2` と同じ順序）、スパイクは `sign(relu(floor(h)))`、リセットは `h − h · s`。すべて SFPU で計算し、ttnn で直接動かすと CPU とビット単位で一致する。2 つの CB を直接足す書き方（`(v + x) * 0.5`）は FPU に回り、fp32 を仮数約 10 ビットに丸める。tt-xla 1.4.0 の経路では、SFPU が読む CB の `UnpackToDestFp32` が引き継がれないので、どう書いても入力を読む時点で丸められる（`discrepancies.md` の D22）
- `make_lif(T)` は `[1, 1, T·R, C]`（R と C は 32 の倍数。時刻 t は `t·R` 行目から）を受ける。tt-mlir は conv の出力を `[1, 1, N·H·W, C]` で置くので、推論モードの BN（conv に畳み込まれる）では、LIF をこの形のまま呼べば、前後に並べ替えも `reshape` も入らない（2 次元の `[T·R, C]` を受けていたときは、前後に `reshape` が 1 つずつ残った。S21）。そのためには、torch の上で LIF の入力 `[T, N, C, H, W]` を `[T·N, C, H, W]` に戻してから `permute(0, 2, 3, 1)` して `[1, 1, T·N·H·W, C]` にする（`xla-sew18.py` の `nhwc4`）。こうすると、tt-mlir が conv の下ろしで入れる `permute` と隣り合い、組で畳まれる。5 次元のまま `permute` すると畳まれない。N·H·W が 32 の倍数でない層（SEW-ResNet18 の batch 8 では layer4 の 392）は、`[1, T, N·H·W, C]` で渡す。タイルは最後の 2 次元で切られるので、時刻ごとの行がタイル単位まで埋まり、時刻の境目がタイルの境目に来る。前後には `reshape` が 1 つずつ残るが、7×7 の面を 32×32 のタイルに置く NCHW を経由しない（S23。それまでは `[T·R, Cc]` に並べ替えていて、1 層あたり約 2 ms かかった）。最後の LIF の出力は `.contiguous()` にしてから平均プーリングに渡す（channels_last の並びのままだと torch_xla が受け付けない）。仕事の単位は 1 時刻ぶんの `(rb, bw)` タイルのブロックで、全コアに 1 次元で分ける（S25。S18〜S24 はタイル 1 行）
- 計算（`make_lif` と `make_lif4`、S25、S27）: h を運び、各時刻、スパイクは `lt(0.99999994, h)`（fp32 で 1 の 1 つ下の値と比べる。h > 0.99999994 と h ≥ 1 は同じ。S25 は `1 − lt(h, 1)`）、次の h は `v + (x − v)·0.5`（v = `h · lt(h, 1)`。`SimpleLIFNode` の `v + (x − v) / 2` とビット単位で同じ）。1 は `fill` で式の中に作る（1.1.10 は `ttl.block.fill`、1.1.5 は `ttl.math.fill` なので、モジュールの頭で別名にする）。tt-lang は `store` ごとに DEST の区間を分け、区間ごとに初期化を繰り返し、共通の部分式も `store` ごとに計算し直すので、`store` の数と式の演算を少なくする形にした（D27）。ブロック（`_blocks`）は、1 コアあたりのタイル数が最小の 10% 以内で、幅、次に高さが最大のもの（上限 32 タイル）。幅 1 タイルの縦長のブロックは遅い。デバイス時間は 20 層で 1.83 ms（S24 のバージョンは 2.81 ms。S27）
- `make_lif4()` は T = 4 の各時刻を別々の `[1, 1, R, C]` で受け、`[1, 1, 4·R, C]` に書く（S24）。stem の conv を時刻ごとの 4 つに分けたとき（`xla-sew18.py` の `STEM_SPLIT=4 STEM_LIF4=1`）、4 つの出力をつながずに読むためのもの。tt-lang はカーネルをソースから組み立て、ループの変数でテンソルを選べないので、4 つの読み出しを書き並べてあり、T = 4 に限る。ブロックと計算は `make_lif` と同じ
- `make_lif2()` は 2 時刻ずつの `[1, 1, 2·R, C]` 2 つを受ける（S32）。stem の conv を 16 枚ずつの 2 つに分けたとき（`STEM_SPLIT=2 STEM_LIF4=1`。8 枚 × 4 より conv が 1.07 → 0.97 ms）に使う。T = 4 に限る
- `make_lif_add(T)` は `make_lif` と同じ形を受け、残差 r（出力と同じ並び）を足した `s + r` を書く（S30）。SEW のブロック（cnf=ADD）の 2 つ目の LIF に使い、`ttnn.add` を消す（`xla-sew18.py` の `LIF_RES=1`。8 層で LIF は 0.23 ms 増え、足し算の 0.71 ms が消える）。値は 0〜4 の整数なので、どの演算器で足しても、tt-xla の経路で CB の読み出しが丸められても正確。DFB が 4 つあるので、ブロックの上限は 8 タイル（layer1 で 220 → 208 µs。S32）
- S14 の古い `make_lif(T)` は `[T·R, Cc]`（R と Cc は 32 の倍数）を受け、2 次元の格子で仕事を分けていた。SEW-ResNet18 では、BN の出力をこの形にする `reshape` と元に戻す `reshape` が LIF ごとに入り（H と W が 32 の倍数でないので、余白を詰め直すデータの移動になる）、1 ステップが 133 ms ではなく 181 ms になった（S16）。tt-lang のカーネルは前後の演算と同じ形で呼ぶ。torch の上で `reshape` や `permute` を足しても、tt-xla 1.4.0 はそれを畳まない（NHWC の 2 次元で呼ぶと `permute` が 28 個増えた）
- tt-xla からの呼び方: `@tt_torch.tt_lang_operation(..., arg_roles=("in", "out"))` の中で、呼ぶたびに `make_lif(T)(x, out)` を作る（同じカーネルのオブジェクトを使い回すと、2 回目の resolve が失敗する。`discrepancies.md` の D18）。tt-lang 1.1.5 で使える演算の制約は D19
- 出力先: カーネルは出力先（`"out"` の引数）に直接書き込む（TTNN では結果を返さない `ttnn.generic`）。出力先を `torch.zeros_like(x)` で作ると、tt-mlir が同じ形の定数を 1 つにまとめるので、同じ形の LIF がすべて 1 つのバッファに書き、2 つの LIF の出力を足すところ（SEW の downsample のあるブロック）では、和が後に書いた方の 2 倍になる。LIF ごとに別の値で埋めた出力先を渡す（`torch.full_like(x, -1.0 - k)`。カーネルが全要素を書き換えるので、値は結果に影響しない。`discrepancies.md` の D25）
- 精度: tt-xla の経路では CB を読むたびに仮数 10 ビットへ切り捨てる（D22。デバイスの出力と CPU でまねた結果がビット単位で一致した）。ただし LIF の入力（conv の出力）がすでに仮数 10 ビットに丸まっているので、SEW-ResNet18 の最初の LIF では、D22 で増える反転は 46 個で、conv の丸めによる CPU との差（約 17,200 個）より 2 桁少ない。D25 を直すと、logits の CPU との相関は元の `SimpleLIFNode` と同じ水準（0.911 と 0.908）になる（S19）

SpikingJelly のモードとの関係（手元の 2.0.0 のコードで確認）は次のとおり。

- single-step（`step_mode='s'`）: 1 回の呼び出しで 1 時刻（`[N, ...]`）を受け、膜電位 `v` はモジュールの状態として次の呼び出しに持ち越す。T ステップ回すには、ネットワーク全体を T 回呼ぶ
- multi-step（`step_mode='m'`）: 1 回の呼び出しで系列全体（`[T, N, ...]`）を受ける。`SimpleLIFNode` は Python で T 回 `single_step_forward` を呼ぶ（`SimpleBaseNode.multi_step_forward`。p150a では 1 ステップごとに約 7 個の TTNN の演算になり、元の遅さの原因だった）。`LIFNode` は `functional.lif_multi_step` で系列全体を 1 回で処理する（A30 では Triton のカーネル）
- このカーネルは `LIFNode` の multi-step と同じ形（系列全体を 1 回で処理し、T のループをカーネルの中に持つ）に当たる。ただし、SpikingJelly の multi-step と次の点が違う

| 点 | SpikingJelly の multi-step | このカーネル |
| --- | --- | --- |
| 最初の膜電位 | 前の呼び出しの最後の `v` を引き継ぐ（`reset_net` で 0 に戻る） | 毎回 0 から始まる（毎回 `reset_net` してから呼ぶのと同じ） |
| 最後の膜電位 | `self.v` に残す。`store_v_seq=True` なら全時刻の `v` も残す | 返さない（L1 の中で捨てる） |
| 学習 | surrogate gradient の backward あり | 推論だけ |
| パラメータ | tau、`v_threshold`、`v_reset`、`decay_input` などを選べる | tau=2、`v_th`=1、`v_reset`=0（hard reset）、`decay_input=True` に固定 |

SEW-ResNet18 のベンチマークは毎ステップ `reset_net` してから系列全体を流すので、この違いは結果に影響しない。長い系列を何回かに分けて流す（前の塊の最後の `v` を次の塊に渡す）には、`v` の初期値の入力と、最後の `v` の出力を足す必要がある（増える DRAM の読み書きは `v` 1 枚ぶん）。single-step の使い方には対応していない。`v` を毎回 DRAM に読み書きするカーネルにすれば作れるが、1 時刻ごとの呼び出しになるので、送り出しの費用が T 倍に戻り、膜電位を L1 に置く利点も消える。single-step と再帰型の SNN の扱いは先送りにした。

