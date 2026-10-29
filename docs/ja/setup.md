# 環境と実行の手順

記録を取った環境は、Supermicro SYS-741GE-TNRT（Ubuntu 22.04、Xeon Max 9468 × 2）に Tenstorrent Blackhole p150a（ファームウェア 19.15.0）と NVIDIA A30 を搭載したもの。パスは環境変数で表記する。`$SNN` は SpikingJelly の clone と venv の配置先、`$VENV_A30`、`$VENV_XLA`、`$VENV_XLA140` は後述の 3 つの venv、`$UV_PY312_LIB` は uv でインストールした Python 3.12 の `lib`、`$DATA` は測定結果の出力先。経緯の番号のうち、D は `discrepancies.md` の番号、S は非公開の作業記録の番号である。

## A30 の環境

公式のインストールガイド（clone の `docs/source/tutorials/en/install.rst`）で検証済みとされている組み合わせに合わせた。

- Python 3.11.16（uv）、torch 2.11.0+cu128、torchvision 0.26.0、Triton 3.6.0
- SpikingJelly は clone から editable モードでインストールした（`uv pip install --editable ".[triton]"`）。ネイティブ CUDA の拡張はまだビルドしていない。

```bash
cd $SNN
uv venv --python 3.11 venv
VIRTUAL_ENV=$PWD/venv uv pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
cd spikingjelly && VIRTUAL_ENV=../venv uv pip install --editable ".[triton]" pytest
```

ニューロンのバックエンドは、テンソルのデバイスに応じて自動で選択される。現在の A30 の環境では、LIF に Triton が選択される。確認方法は次のとおり。

```bash
../venv/bin/python -c 'import torch; from spikingjelly.activation_based import functional; print(functional.neuron_implementation("lif", torch.device("cuda:0")))'
```

## A30 での計測

`benchmark/benchmark_snn_single_gpu.py` を用いる。入力が乱数であるため、速度の比較のみを目的としたもので、精度は評価できない。clone の直下で実行する。

```bash
../venv/bin/python benchmark/benchmark_snn_single_gpu.py case --model sew_resnet18 \
  --phase inference --execution eager --batch-size 32 --warmup 10 --steps 30 \
  --require-gpu-name A30 --output $DATA/a30_sew18_inference_eager_bs32_T4.json
```

モデルは `spiking_vgg16_bn`、`sew_resnet18`、`spikformer_ti`、`spikformer_s` から選択できる。`--execution` は `eager`、`compile`、`cuda_graph` から選択する。

2026-10-27 の結果（SEW-ResNet18、LIF、fp32、224×224、T=4、batch 32、Triton、eager）。

| phase | 中央値 | p10〜p90 | ピークのメモリ |
| --- | --- | --- | --- |
| 推論 | 51.1 ms | 51.0〜51.4 ms | 1.57 GB |
| 学習（SGD） | 169.5 ms | 168.8〜170.0 ms | 6.25 GB |

ピークのメモリは `peak_allocated_bytes`（10^9 B で換算）。

ベンチマークは TF32 を無効にする（`benchmark_snn_single_gpu.py` の 502〜503 行目）ため、conv は厳密な fp32 で計算される。p150a の conv の出力は仮数 10 ビット相当であるため、精度を揃えて比較するときは、TF32 を有効にした時間を `tools/a30-sew18.py` で測定する（`discrepancies.md` の D26）。2026-10-28 の結果（SEW-ResNet18、`LIFNode`、推論モードの BN、224×224、T=4、batch 8、eager。S20）。

| 条件 | 1 ステップ |
| --- | --- |
| ベンチマークのまま（厳密な fp32） | 14.75 ms（cuda_graph 14.58 ms、compile 19.76 ms） |
| TF32 を有効にし、`cudnn.benchmark=True` | 7.5 ms |

TF32 を有効にすると、ウォームアップの後も 12〜20 ステップは約 10 ms と遅いため、それより多くのステップを実行してから測定する。p150a との比較の基準は TF32 の値とする（ユーザーの決定、2026-10-28。ベンチマークのままの値は併記する）。

## p150a（torch_xla）の環境

- venv: `$VENV_XLA`（Python 3.12.14、`pjrt-plugin-tt==1.5.0.dev20260831000501`、torch 2.11.0+cpu、torch-xla 2.9.0。SpikingJelly は `--no-deps -e` でインストール）
- venv（1.4.0）: `$VENV_XLA140`（2026-10-27。`venv-xla` と同じ構成で、`pjrt-plugin-tt==1.4.0`、torch-xla 2.9.0+git56f8e61。1.5.0.dev20260831 では level 1・2 で Conv をコンパイルできないため。S8・S10）
- 実行時に `LD_LIBRARY_PATH=$UV_PY312_LIB` を指定する（torch_xla が libpython を見つけられないため）。

```bash
LD_LIBRARY_PATH=$UV_PY312_LIB \
  $VENV_XLA/bin/python tools/xla-cnn-step2.py 5 lif manual
```

p150a で使用するときの制約（2026-10-27、S2）は次のとおり。

- `LIFNode` などの登録済みのニューロンは、CPU と CUDA の実装しかないため、xla では実行できない。純粋な PyTorch の `SimpleLIFNode` を用いる。
- Conv の直後の `nn.BatchNorm2d`（学習モード）は、backward が正しく計算されない。`tools/sj_xla.py` の `ManualBN2d` を用いる。eval モードの BN は、forward ならそのまま実行できる（optimization_level 0）。
- SEW-ResNet18 は batch 12 以上で、実行時にエラーで停止する（`Invalid subtile broadcast type`。1.5.0 は 16 で確認、1.4.0 は 12 と 16 で確認）。
- optimization_level 1・2 は、Conv を含むとコンパイルできない（1 は `ttnn.permute failed validation`、2 はコンパイラの segfault）。Conv 1 個でも、bf16 でも、公式の例 `examples/pytorch/mnist_performant.py` でも同じ（S6）。1.5.0.dev20260831 の退行で、1.4.0（`venv-xla140`）なら level 1 でコンパイルできる（S8、S10）。1.4.0 でも level 2 はコンパイルが終了しない。
- trace（`enable_trace`）は、小さいモデル（Conv＋BN＋ReLU）では level 0〜2 のどれでもコンパイル時に abort する（`Memory layout is required for non-SystemMemory buffer type`。Linear のみなら動作する。S6、S8）。ただし SEW-ResNet18（level 1、推論モードの BN）は trace で動作し、ホストのディスパッチのコスト（trace なしで 1 ステップ約 22〜27 ms）がほぼなくなる（`xla-sew18.py` の `trace` の引数。S23、D15）。
- optimization_level 1 は、入力が 3〜16 チャネルの stem の conv（SEW-ResNet18 の 7×7、stride 2）を誤って計算する（CPU との相関 0.28。32 チャネルにゼロパディングすると正しく計算される。ほかの conv は正しい）。`xla-sew18.py` は `STEM_PAD=32` で回避する（`discrepancies.md` の D24）。
- SpikingJelly のベンチマークの推論は、モデルを推論モードにして実行する（`model.train(False)`）。p150a でも BN は推論モードで測定する（`xla-sew18.py` の `BN=eval`。D23）。推論モードの BN は、tt-mlir が conv に統合する（BN folding）。
- 時間の測定方法: `torch_xla.sync(wait=True)` はデバイスの処理の完了を待機しない。1 ステップの時間は、出力を `.cpu()` でホストへ転送するまでを測定する（`xla-sew18.py` と `xla-ttl-lif.py` の `WAIT=cpu`。`discrepancies.md` の D21）。
- 1 ステップごとに同期するときの時間は、ホストの CPU の省電力の動作により 3〜5 ms 延び、実行ごとにばらつく。待機のたびにスレッドがスリープし、CPU が 800 MHz（`acpi-cpufreq` と `schedutil` の最低値）のまま動作するため。SMT の兄弟の論理 CPU（156〜167）に SCHED_IDLE のビジーループ（`taskset -c <cpu> chrt -i 0 python3 -c 'while 1: pass'`）を置き、計測のスレッドを CPU 60〜71 に固定すると解消する（SEW-ResNet18 で 14.2 → 11.0 ms）。まとめて同期にはほとんど効果がない（9.59 → 9.42 ms）。既定の値は省電力の設定のまま測定する（S31、D31）。
- tt-xla 1.4.0 の trace の入力は、常にホストの行優先のレイアウトとして扱われ、trace の中でタイルのレイアウトに変換される。デバイス上のタイルのレイアウトのテンソルを渡しても同じ（S33、D32）。

| スクリプト | 内容 |
| --- | --- |
| `tools/xla-op-backward.py` | 演算子ごと（とその組み合わせ）の backward を CPU と比較する。`ONLY=` で対象を選択し、`OPTS=` で tt-xla のコンパイルオプションを渡す |
| `tools/xla-bn-eval.py [lazy\|compile] [level]` | Conv → eval の BN の forward を、lazy tensor と `torch.compile(backend="tt")` で CPU と比較する |
| `tools/xla-sew18.py [compile\|lazy] [batch] [T] [steps] [size] [level] [trace\|notrace] [perf_metrics_file]` | SEW-ResNet18 の推論を SpikingJelly のベンチマークと同じ構成（ニューロンのみ `SimpleLIFNode`）で測定し、logits を CPU と比較する。`NEURON=ttl` で LIF を `ttl_lif.py` のカーネルに置換する（`venv-xla140`）。LIF に渡すテンソルの形状と次元の順序は `LIF_LAYOUT` で選択する。推論モードの BN では `nhwc4`（conv の出力の順序のまま `make_lif` に渡す）、学習モードの BN では `nchw`（BN の出力の形状のまま `make_lif_nchw` に渡す）。既定の `flat`（`[T·R, Cc]` に並べ替える）と `nhwc` は不利。`NEURON=none` で恒等写像に置換し（CPU の基準も恒等写像）、`NEURON=reshape` で `flat` と同じ並べ替えの間に `x * 0.5` を挟む。`BN=eval` で推論モード（ベンチマークと同じ）、`BN=none` で BN を除去する。`STEM_PAD=32` で stem の conv の入力チャネルをゼロパディングする（D24）。`STEM_FOLD=1` で、入力をホストで 2×2 の space-to-depth で変換し、stem を 4×4、stride 1 の conv にする（`STEM_PAD=32` と組み合わせて使う。16 チャネルでは D24 が起きる）。`INPUT_LAYOUT=nhwc` で、入力をホストで `[1, 1, T·N·H·W, C]` に並べ替えてからデバイスに配置する（S23）。`STEM_SPLIT=k`（`BN=eval` と組み合わせて使う）で、stem の BN をホストで conv に統合し、T·N 枚を k 組に分割して k 個の conv で計算する（S24。32 枚のままでは出力が L1 に収まらず、tt-mlir が DRAM を経由して分割処理する）。`INPUT_LAYOUT=nhwc` と併用すると、入力は組ごとの k 個の `[1, 1, b·H·W, C]` になる。`STEM_LIF4=1`（`STEM_SPLIT=4` か `2`、T = 4、`NEURON=ttl` と組み合わせて使う）で、stem の LIF が 4 つ（`make_lif4`）か 2 つ（`make_lif2`。S32）の conv の出力を直接ロードする（連結が不要になる）。現在の最良は `STEM_SPLIT=2`（16 枚の conv 2 つ。8 枚 × 4 より 0.1 ms 速い）。`LIF_RES=1`（`LIF_LAYOUT=nhwc4` と組み合わせて使う）で、SEW のブロックの残差結合の加算を 2 つ目の LIF のカーネルと融合する（`make_lif_add`。S30）。`AVGPOOL=matmul` で、最後の平均プーリングを 0/1 の行列との積に置換する（並べ替えが不要になる。S34）。現在の最良の組み合わせは README_ja.md の「再現手順」に記載している。`LIF_GRID=11,10` で `make_lif` と `make_lif4` を p150a の 110 コアで実行する（指定しない場合は、tt-xla の tt-lang 統合が渡す仮の device に合わせて 64 コアになる。D28）。`STEM_SPIKE=bf16`（`STEM_LIF4=1` と組み合わせて使う）で、stem の LIF がスパイクを bf16 で書き込む（S26。tt-xla の経路ではデバイスがハングした。D29。使用しない）。`WAIT=cpu` で毎ステップ logits をホストへ転送する（時間を測定するときは必ず指定する）。`PREFIX=<段>`（`conv1`、`bn1`、`sn1`、`maxpool`、`layer1`〜`layer4`）でその段までのみを実行し、出力を CPU と要素ごとに比較する。`LIF_OUT=zeros` で、LIF の出力先が共有される元の実装に戻す（D25 の再現用）。`SAVE_LOGITS=<file.pt>` で 1 ステップ目の logits と CPU の基準を保存する。`PAUSE=<step>:<秒>[:spin]` で指定したステップの前に一時停止し、`EXPORT=<dir>` で IR を出力する。各ステップの時刻、CPU 時間、RSS と、まとめて同期での送信の時間とデバイスの処理が追いつくまでの待機の時間（`pipelined_split`）も出力する |
| `tools/lif-d22-emul.py [trunc\|rne] [batch] [T] [size] [saved_logits.pt ...]` | CPU のみで、tt-xla の経路の tt-lang の LIF（CB を読み出すたびに仮数 10 ビットへ切り捨てる。D22）をエミュレートして SEW-ResNet18 を実行し、層ごとのスパイクの反転と logits を正確な LIF と比較する。`xla-sew18.py` の `SAVE_LOGITS` で保存したデバイスの logits とも比較する |
| `tools/xla-conv-check.py <level> <module path> [spikes\|randn]` | SEW-ResNet18 の conv を 1 つだけ、デバイスと CPU で比較する（D24 の確認） |
| `tools/xla-stem-pad.py <level> <cin> [batch]` | stem の conv の入力チャネルを cin までゼロパディングし、デバイスと CPU で比較する（D24 が修正されたかの確認） |
| `tools/xla-op-hostcost.py <level> <add\|lif\|lif_alg\|conv\|bn\|conv_bn_lif> <K...>` | 同じ演算を K 個連結した小さいグラフで、演算 1 個あたりのホストの時間を測定する。`lif_alg` は演算数を削減した LIF の式。`OPTS=` でコンパイルオプションを追加し、`PERF=` で演算の数を出力する。`SHAPE=N,C,H,W` で実際のサイズを指定し、`WAIT=cpu` で出力をホストへ転送してデバイスの処理の完了までを測定する（まとめて同期の時間と、その送信の時間とデバイスの処理が追いつくまでの待機の時間も出力する）。`REDUCE=1` で出力の和を取り、ホストへの転送量を小さくする（S23） |
| `tools/ttl_lif.py [T] [R] [C] [fp32] [split4 [bf16out] \| split2 \| res]` | LIF（`SimpleLIFNode` と同じ式）の T ステップをまとめて計算する tt-lang のカーネル。`make_lif` は `[1, 1, T·R, C]`（conv の出力の順序）と `[1, T, R, C]`（R が 32 の倍数でないとき）、`make_lif4` は T = 4 の各ステップを別々の `[1, 1, R, C]` で（`split4` で単体の試験。`bf16_out=True`（`bf16out`）でスパイクを bf16 で書き込む。S26）、`make_lif2` は 2 時刻ずつの `[1, 1, 2·R, C]` 2 つで（`split2`。S32）、`make_lif_nchw` は `[T·N, C, H, W]`（BN の出力の順序）を受け取る。`make_lif_add` は `make_lif` に残差 r の加算を融合したもので、`s + r` を書き込む（`res`。S30）。単体で起動すると、ttnn で実機上で実行し、CPU と比較する（コンテナで `run-test.sh --hw`。しきい値ちょうどの例も確認する）。fp32 のみ。tt-lang 1.1.5 と 1.1.10 の両方で動作する記述にしてある（`discrepancies.md` の D19） |
| `tools/ttl-lif-bench.py <kernel.py>:<factory>:<flat\|nchw\|nhwc\|nhwcres\|split4\|split4bf16\|split2>[,...] [sew18\|B:C:H:W,...] [reps]` | コンテナで、LIF のカーネルのデバイス時間を、デバイスのプロファイラのカーネルの区間で形状ごとに測定する。複数のバージョンを同じ入力で実行し、出力がビット単位で一致するかも確認する。`split4` は `make_lif4` を 4 入力で測定する（`factory()` を T なしで呼び出す）。`split4bf16` は bf16 の出力で測定する（`factory(bf16_out=True)`）。`split2` は `make_lif2` を 2 入力で、`nhwcres` は `make_lif_add` を残差（0〜3 の整数）つきで測定する（S30、S32）。プロファイラは CB の操作ごとに計測を追加するので、小さなブロックを多数処理する形状では時間が過大に出る（S16）。`LIF_MEM=l1`（L1 の interleaved）、`l1h`（11×10 の格子に行方向で分割した L1。height sharding）で、入出力を L1 に配置して測定する（S29） |
| `tools/ttnn-replay-ir.py <ttnn.mlir> <out.json> [reps] [T]` | コンテナで、tt-xla が出力した TTNN の IR の主関数の演算を 1 つずつ ttnn で再実行し、演算ごとのデバイス時間（稼働と区間）をプロファイラで測定する（tt-xla 1.4.0 ではプロファイラを使用できないため。コンテナの tt-metal はバージョンが異なるので目安の値）。trace ありで出力した IR では、trace 用の関数（`@trace_0_main`）を再実行する。IR はホームディレクトリの下にコピーして渡す（コンテナからは `/work` と `/tmp` を参照できない。`~/bench-data` も `/work` へのリンクなので参照できない。パスは `$HOME` を用いた絶対パスで指定する）。`ttnn.generic` には、入力の数と形状に応じて `make_lif`、`make_lif4`、`make_lif_add`、`make_lif2` を割り当てる（S30、S32）。`[1, 4, 392, 512] → [4, 392, 512]` の `reshape`（`AVGPOOL=matmul`）は、コンテナの ttnn では実行できない（S34） |
| `tools/ttnn-replay-chain.py <ttnn.mlir> <out.json> <policy> [reps] [T]` | コンテナで、tt-xla が trace ありで出力した IR の trace の関数を、各演算の出力を次の演算の入力として連鎖的に再実行し（IR の `deallocate` も反映する）、演算ごとのデバイス時間を測定する。出力の配置先はポリシーで選択する（`mem=dram\|l1\|l1s`、`l1_from`、`dbuf`、`lif_tiles`、`dram_ops`。スクリプトの冒頭に一覧）。L1 に収まらない演算は、出力の配置先を L1 の interleaved、DRAM の順にフォールバックする。それでも実行できなければ、L1 にあるテンソルをすべて DRAM へ移動してから、出力を DRAM に配置して実行する。最初の 1 回で LIF のスパイクを CPU と照合する（S29） |
| `tools/chain-csv.py <out dir> <key>::<label>::<chain.json> ...` | `ttnn-replay-chain.py` の結果を、時間の内訳のダッシュボード（含めていない）が読み込む `l1policies.csv`（配置ポリシーごとの 1 ステップの稼働と、L1 からテンソルを移動した演算の数）と `l1chain.csv`（配置ポリシーごと、種類ごとの稼働。種類の分類は `breakdown-sew18.py` と同じ）に変換する（S29） |
| `tools/a30-sew18-profile.py [LIFNode\|SimpleLIFNode] [batch] [T] [size] [steps] [out.json]` | A30 で、層ごとの GPU の時間を torch.profiler で測定する（`a30-sew18.py` と同じ構成とスイッチ） |
| `tools/breakdown-sew18.py <replay.json> <a30 tf32.json> <a30 fp32.json> <out.json> [<比べる前の replay.json> <csv dir>]` | `ttnn-replay-ir.py` と `a30-sew18-profile.py` の結果を種類ごと、層ごとの内訳に集計し、時間の内訳のダッシュボード（含めていない）が読み込む 8 つの CSV を出力する。比較用に以前の replay を与えると、その結果が「p150a（始める前）」の行になる。実測した 1 ステップの時間と高速化の経過は、スクリプト内の `STEPS` と `HISTORY` に記載している。1 ステップの下限の見積もり（`bounds.csv` と前提の値の `bound_params.csv`。S28）も出力する。前提（DRAM の帯域、演算性能）はスクリプト内の定数 |
| `tools/a30-sew18.py [LIFNode\|SimpleLIFNode] [batch] [T] [size] [steps]` | A30 で SEW-ResNet18 の推論を `xla-sew18.py` と同じ構成で測定する。`TF32`、`CUDNN_BENCH`、`BN`、`GRAD` で、ベンチマークと `a30-hostcost.py` で異なる条件を 1 つずつ切り替え、CUDA event、1 ステップごとに同期、まとめて同期の 3 通りの計時を出力する |
| `tools/sj_xla.py` | p150a 用の回避策（`ManualBN2d`） |
| `tools/thread-sampler.py <pid> [interval] [duration]` | 実行中のプロセスについて、スレッドごとの CPU 時間、最後に実行された CPU、その CPU の周波数を記録する（読み取りのみ） |

## LIF の融合カーネル（tt-lang、2026-10-27）

`tools/ttl_lif.py` の `make_lif(T)` は、SpikingJelly の `SimpleLIFNode`（tau=2、`decay_input=True`、`v_threshold=1`、`v_reset=0` の hard reset、`detach_reset=True`）の T ステップを、1 つの tt-lang のカーネルで計算する。推論のみに対応し、fp32 を前提とする（S14）。

- 入出力: `make_lif_nchw(T)` は `[T·N, C, H, W]`（BN の出力の形状のまま。H と W はタイルにパディングを加えて配置されている）を受け取る。時刻 t の画像 n は `t·N + n` 番目なので、各 (n, c) の面は別々のタイルに配置され、時刻の境界は必ずタイルの境界と一致する。1 時刻ぶんの N·C 枚の面を全コアに均等に分割し、各コアが担当する面について T ステップを順に処理し、膜電位は DFB（L1）に保持したまま次の時刻へ持ち越す。パディングの要素も計算する（読み出されない）。
- 計算（`make_lif_nchw`）: h₀ = `x * 0.5`、hₜ = `v + (x * 0.5 − v * 0.5)`（`SimpleLIFNode` の `v + (x − v) / 2` と同じ順序）、スパイクは `sign(relu(floor(h)))`、リセットは `h − h · s`。すべて SFPU で計算し、ttnn で直接実行すると CPU とビット単位で一致する。2 つの CB を直接加算する記述（`(v + x) * 0.5`）は FPU で計算され、fp32 を仮数約 10 ビットに丸める。tt-xla 1.4.0 の経路では、SFPU が読み出す CB の `UnpackToDestFp32` の設定が引き継がれないため、どう記述しても入力を読み出す時点で丸められる（`discrepancies.md` の D22）。
- `make_lif(T)` は `[1, 1, T·R, C]`（R と C は 32 の倍数。時刻 t は `t·R` 行目から）を受け取る。tt-mlir は conv の出力を `[1, 1, N·H·W, C]` で配置するので、推論モードの BN（conv に統合される）では、LIF をこの形状のまま呼び出せば、前後に並べ替えも `reshape` も挿入されない（2 次元の `[T·R, C]` を受け取っていたときは、前後に `reshape` が 1 つずつ残った。S21）。そのためには、torch の上で LIF の入力 `[T, N, C, H, W]` を `[T·N, C, H, W]` に戻してから `permute(0, 2, 3, 1)` で `[1, 1, T·N·H·W, C]` にする（`xla-sew18.py` の `nhwc4`）。こうすると、tt-mlir が conv の lowering で挿入する `permute` と隣接し、対で相殺されて除去される。5 次元のまま `permute` すると除去されない。N·H·W が 32 の倍数でない層（SEW-ResNet18 の batch 8 では layer4 の 392）には、`[1, T, N·H·W, C]` で渡す。タイルは最後の 2 次元で区切られるので、時刻ごとの行がタイル単位までパディングされ、時刻の境界がタイルの境界と一致する。前後には `reshape` が 1 つずつ残るが、7×7 の面を 32×32 のタイルに配置する NCHW を経由しない（S23。それまでは `[T·R, Cc]` に並べ替えていて、1 層あたり約 2 ms かかっていた）。最後の LIF の出力は `.contiguous()` にしてから平均プーリングに渡す（channels_last のメモリレイアウトのままでは torch_xla が受け付けない）。処理の単位は 1 時刻ぶんの `(rb, bw)` タイルのブロックで、全コアに 1 次元で分配する（S25。S18〜S24 はタイル 1 行）。
- 計算（`make_lif` と `make_lif4`、S25、S27）: 次の時刻へは h を持ち越す。各時刻のスパイクは `lt(0.99999994, h)`（fp32 で 1 の 1 つ下の値と比較する。h > 0.99999994 と h ≥ 1 は同じ。S25 は `1 − lt(h, 1)`）、次の h は `v + (x − v)·0.5`（v = `h · lt(h, 1)`。`SimpleLIFNode` の `v + (x − v) / 2` とビット単位で同じ）。定数の 1 は `fill` で式の中に生成する（1.1.10 は `ttl.block.fill`、1.1.5 は `ttl.math.fill` なので、モジュールの先頭で別名を定義する）。tt-lang は `store` ごとに DEST の区間を分け、区間ごとに初期化を繰り返し、共通の部分式も `store` ごとに再計算するため、`store` の数と式の演算数を削減する形にした（D27）。ブロック（`_blocks`）は、1 コアあたりのタイル数が最小値から 10% 以内のもののうち、幅、次に高さが最大のもの（上限 32 タイル）。幅 1 タイルの縦長のブロックは遅い。デバイス時間は 20 層で 1.83 ms（S24 のバージョンは 2.81 ms。S27）。
- `make_lif4()` は T = 4 の各時刻を別々の `[1, 1, R, C]` で受け取り、`[1, 1, 4·R, C]` に書き込む（S24）。stem の conv を時刻ごとの 4 つに分割したとき（`xla-sew18.py` の `STEM_SPLIT=4 STEM_LIF4=1`）、4 つの出力を連結せずに読み出すためのもの。tt-lang はカーネルをソースから構築し、ループの変数でテンソルを選択できないため、4 つの読み出しを並べて記述してあり、T = 4 に限る。ブロックと計算は `make_lif` と同じ。
- `make_lif2()` は 2 時刻ずつの `[1, 1, 2·R, C]` 2 つを受け取る（S32）。stem の conv を 16 枚ずつの 2 つに分割したとき（`STEM_SPLIT=2 STEM_LIF4=1`。8 枚 × 4 より conv が 1.07 → 0.97 ms）に用いる。T = 4 に限る。
- `make_lif_add(T)` は `make_lif` と同じ形状を受け取り、残差 r（出力と同じレイアウト）を加算した `s + r` を書き込む（S30）。SEW のブロック（cnf=ADD）の 2 つ目の LIF に用い、`ttnn.add` を排除する（`xla-sew18.py` の `LIF_RES=1`。8 層で LIF は 0.23 ms 増え、加算の 0.71 ms がなくなる）。値は 0〜4 の整数なので、どの演算器で加算しても、tt-xla の経路で CB の読み出しが丸められても正確。DFB が 4 つあるため、ブロックの上限は 8 タイル（layer1 で 220 → 208 µs。S32）。
- S14 の古い `make_lif(T)` は `[T·R, Cc]`（R と Cc は 32 の倍数）を受け取り、2 次元の格子で処理を分割していた。SEW-ResNet18 では、BN の出力をこの形状にする `reshape` と元に戻す `reshape` が LIF ごとに挿入され（H と W が 32 の倍数でないため、パディングを詰め直すデータの移動になる）、1 ステップが 133 ms ではなく 181 ms になった（S16）。tt-lang のカーネルは前後の演算と同じ形状で呼び出す。torch の上で `reshape` や `permute` を追加しても、tt-xla 1.4.0 はそれを除去しない（NHWC の 2 次元で呼び出すと `permute` が 28 個増えた）。
- tt-xla からの呼び出し方: `@tt_torch.tt_lang_operation(..., arg_roles=("in", "out"))` の中で、呼び出すたびに `make_lif(T)(x, out)` を生成する（同じカーネルのオブジェクトを再利用すると、2 回目の resolve が失敗する。`discrepancies.md` の D18）。tt-lang 1.1.5 で使用できる演算の制約は D19。
- 出力先: カーネルは出力先（`"out"` の引数）に直接書き込む（TTNN では結果を返さない `ttnn.generic`）。出力先を `torch.zeros_like(x)` で生成すると、tt-mlir が同じ形状の定数を 1 つにまとめるため、同じ形状の LIF がすべて 1 つのバッファに書き込み、2 つの LIF の出力を加算する箇所（SEW の downsample のあるブロック）では、和が後に書き込んだ方の 2 倍になる。LIF ごとに別の値で埋めた出力先を渡す（`torch.full_like(x, -1.0 - k)`。カーネルが全要素を書き換えるので、値は結果に影響しない。`discrepancies.md` の D25）。
- 精度: tt-xla の経路では CB を読み出すたびに仮数 10 ビットへ切り捨てる（D22。デバイスの出力と、CPU でエミュレートした結果がビット単位で一致した）。ただし LIF の入力（conv の出力）がすでに仮数 10 ビットに丸められているため、SEW-ResNet18 の最初の LIF では、D22 で増える反転は 46 個で、conv の丸めによる CPU との差（約 17,200 個）より 2 桁少ない。D25 を修正すると、logits の CPU との相関は元の `SimpleLIFNode` と同じ水準（0.911 と 0.908）になる（S19）。

SpikingJelly のモードとの関係（手元の 2.0.0 のコードで確認）は次のとおり。

- single-step（`step_mode='s'`）: 1 回の呼び出しで 1 時刻（`[N, ...]`）を受け取り、膜電位 `v` はモジュールの状態として次の呼び出しに持ち越す。T ステップを処理するには、ネットワーク全体を T 回呼び出す。
- multi-step（`step_mode='m'`）: 1 回の呼び出しで系列全体（`[T, N, ...]`）を受け取る。`SimpleLIFNode` は Python で T 回 `single_step_forward` を呼び出す（`SimpleBaseNode.multi_step_forward`。p150a では 1 ステップごとに約 7 個の TTNN の演算になり、元の遅さの原因だった）。`LIFNode` は `functional.lif_multi_step` で系列全体を 1 回で処理する（A30 では Triton のカーネル）。
- このカーネルは `LIFNode` の multi-step と同じ形式（系列全体を 1 回で処理し、T のループをカーネルの中に持つ）に相当する。ただし、SpikingJelly の multi-step とは次の点が異なる。

| 点 | SpikingJelly の multi-step | このカーネル |
| --- | --- | --- |
| 最初の膜電位 | 前の呼び出しの最後の `v` を引き継ぐ（`reset_net` で 0 に戻る） | 毎回 0 から始まる（毎回 `reset_net` してから呼び出すのと同じ） |
| 最後の膜電位 | `self.v` に残す。`store_v_seq=True` なら全時刻の `v` も残す | 返さない（L1 の中で破棄する） |
| 学習 | surrogate gradient の backward あり | 推論のみ |
| パラメータ | tau、`v_threshold`、`v_reset`、`decay_input` などを選択できる | tau=2、`v_th`=1、`v_reset`=0（hard reset）、`decay_input=True` に固定 |

SEW-ResNet18 のベンチマークは毎ステップ `reset_net` してから系列全体を処理するので、この違いは結果に影響しない。長い系列を複数回に分割して処理する（前のチャンクの最後の `v` を次のチャンクに渡す）には、`v` の初期値の入力と、最後の `v` の出力を追加する必要がある（増える DRAM の読み書きは `v` 1 枚ぶん）。single-step の使い方には対応していない。`v` を毎回 DRAM に読み書きするカーネルにすれば実装できるが、1 時刻ごとの呼び出しになるため、ディスパッチのコストが T 倍に戻り、膜電位を L1 に保持する利点もなくなる。single-step と再帰型の SNN の扱いは先送りにした。

