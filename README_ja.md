# Spiking ResNet on Tenstorrent

[English](README.md)

SpikingJelly 2.0 の SEW-ResNet18 の推論を、Tenstorrent Blackhole p150a で実装して高速化した記録。モデルは tt-xla（torch_xla と PJRT プラグインの `pjrt-plugin-tt`）でコンパイルし、LIF ニューロンだけを tt-lang で記述したカーネルに置換した。

- 推論 1 ステップ（T = 4 時刻ぶん）の時間: 222.8 ms → 9.44 ms
- NVIDIA A30（TF32）では同じ 1 ステップが 7.52 ms。p150a の時間はその 1.26 倍。
- ただし、p150aにはまだ多くの最適化の余地があり、帯域を最大限活用できれば、実行時間は現在のおよそ半分まで減少する見込みである。例えば、現状はConvやLinearへのカーネル最適化を行っておらず、コンパイラ任せである。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/step-times-dark.png">
  <img alt="変更前と変更後の 1 ステップの時間。変更前（元の SimpleLIFNode）は 222.8 ms、すべての変更を入れた後は 9.44 ms で 23.6 倍速く、A30（TF32）の 7.52 ms に近づく。途中の棒は変更を 1 つずつ積み重ねた時間" src="docs/figures/step-times-light.png" width="800">
</picture>

灰色の棒が変更前、濃い青の棒がすべての変更を入れた後。途中の棒は、それより上の変更をすべて含む。

## 動機と手法、主張範囲
Tenstorrent Blackholeアーキテクチャは、格子状に並べたコアを2次元トーラスのNoC（2系統）でつないだ分散メモリのメニコアAIアクセラレータであり、この特徴はNeuromorphic ProcessorであるIntel Loihi/Loihi2に類似している。
そのため、Spiking Neural Networks (SNN) の高速化に向くと考え、まずは最も広く使われているSNNライブラリの一つであるSpikingJellyで実装された最も有名なモデルの一つであるSEW-ResNetの高速化に取り組む。
そのうえで、Leaky Integrate-and-Fire Neuronの専用Fused Kernelを作る以外は、通常のTenstorrent hardware向け機械学習フレームワークコンパイラtt-xlaの最適化以外を出来るだけ用いずに高速化したい。ただし、PyTorch側でのモデルの書き換え（stemのconvのspace-to-depthによる変換と分割、平均プーリングの行列積への置換、テンソルの次元の順序の変更）と、ホストでの入力の前処理は用いた。
Tenstorrentはエージェントによる最適化を支援するためのskillを提供しているため、これをClaude CodeでOpus5.5が用いることで、半自動で高速化に取り組む。
Tenstorrent BH p150aへのSNNの実装において、エージェントによるパフォーマンスエンジニアリングがどの程度機能するかのテストであるため、厳密な他種の計算基盤との比較などは現状意図していない。
TenstorrentハードウェアとSNNの相性について判断するためにはもっと適したSNNの構成が存在するはずだが、まずは有名なものでテストしているため、それについても主張はしない。

## 結果

条件: SEW-ResNet18（残差結合は ADD を用いる）、入力 224 × 224、batch 8、T = 4、fp32、BatchNorm は推論モード。

- まとめて同期: k ステップを続けて送信し、最後に 1 回だけ待機する。かかった時間を 1 ステップあたりに換算した値。
- 1 ステップごとに同期: 毎ステップ、logits をホストへ転送する。

| 段階 | まとめて同期 | 1 ステップごとに同期 |
| --- | --- | --- |
| 元の `SimpleLIFNode`（LIF を TTNN の小さな演算で実装していた） | 222.8 ms | 255.8 ms |
| LIF を tt-lang の単一カーネルに置換する（BN の出力の順序 NCHW で呼び出す） | 70.1 ms | 85.7 ms |
| LIF のカーネルを conv の出力の順序で呼び出す | 31.5 ms | 43.0 ms |
| LIF の前後の reshape を排除する。layer4 の LIF には `[1, T, R, C]` の形状で入力する | 28.3 ms | 34.6 ms |
| trace を用いる | 21.1 ms | 27.2 ms |
| 入力の順序をホストで conv と揃える | 17.1 ms | 22.7 ms |
| stem の conv を、2 × 2 の space-to-depth で stride 1 の conv に変換する | 14.5 ms | 19.6 ms |
| stem の conv を時刻ごとに分割し、後段の LIF が直接ロードする | 11.9 ms | 17.3 ms |
| LIF のカーネルの計算を削減し、使用コアを 110 に増やす | 10.1 ms | 15.5 ms |
| 残差結合の加算を LIF と融合させる | 9.59 ms | 14.2 ms |
| stem の conv を 16 枚 × 2 に分割し、平均プーリングを行列積に置換する | **9.44 ms** | 15.0 ms |
| （参考）A30、TF32 と `cudnn.benchmark` | 7.52 ms | 7.63 ms |
| （参考）A30、SpikingJelly のベンチマークと同じ厳密な fp32 | 14.75 ms | — |

変更前と変更後の、1 ステップの時間の内訳は次のとおり。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/breakdown-dark.png">
  <img alt="1 ステップの内訳。p150a の変更前は並べ替えなど 15.78 ms、プーリングなど 3.96 ms、stem の conv 3.88 ms、conv 3.71 ms、LIF 3.03 ms。変更後は conv 3.71 ms、LIF 2.11 ms、並べ替えなど 1.30 ms、プーリングなど 1.11 ms、stem の conv 0.96 ms。A30 は conv 3.25 ms、LIF 1.54 ms、stem の conv 1.06 ms、BN 0.97 ms、プーリングなど 0.58 ms" src="docs/figures/breakdown-light.png" width="800">
</picture>

変更前の棒は、内訳を最初に測定した段階（LIF をカーネル化し、conv の出力の順序で呼び出した段階。表の 3 行目で、測定し直した 1 ステップは 31.39 ms）。元の `SimpleLIFNode` の段階（222.8 ms）はホストのディスパッチが律速しており、IR の再実行による内訳は測定していない。

p150a の値は、tt-xla が出力した TTNN の IR を演算ごとにデバイスで再実行し、測定した稼働時間の和（変更後 9.19 ms、実測の 1 ステップは 9.44 ms。変更前 30.36 ms）。再実行にはコンテナ内の tt-metal を用いた。tt-xla が用いる tt-metal とはバージョンが異なるため、目安の値である。A30 の値は torch.profiler で測定した GPU のカーネル時間の和（7.41 ms。実測は 7.52 ms）。p150a では、BN は conv に統合され、残差結合の加算は LIF と融合されている。

### 結果の解釈上の注意

- 測定したのは速度のみ。重みは初期値のままで、入力は乱数（SpikingJelly のベンチマークと同じ）。学習済みの重みでの精度は評価していない。
- 正確性は、同じプロセスで CPU と層ごとに比較して確認した。数学的に等価なはずの変更（並べ替えの排除、分割、融合）では、logits がビット単位で一致することを確認した。
- p150a の conv の出力の精度は仮数 10 ビット程度であるため、しきい値の近傍でスパイクが反転する。LIF をカーネル化した時点で、デバイスの logits と CPU の logits の相関は 0.911（デバイス上の元の `SimpleLIFNode` は 0.908）。CPU のみで各層の出力に丸め誤差程度の摂動を加えても、相関は 0.92 前後で下げ止まる。
- 後半の段階では、入力の次元の並べ替え、チャネルのゼロパディング、space-to-depth、時刻ごとの分割をホストで処理している（trace 使用時で約 6.6 ms）。この時間は p150a の値に含まれない。
- A30 の値は、同じモデルを同じ構成で実行して測定したもの（LIF は SpikingJelly の `LIFNode` で、A30 では Triton のカーネルとして実行される）。SpikingJelly のベンチマーク自体は TF32 を無効にするため、両方の値を示した。
- バージョン: `pjrt-plugin-tt` 1.4.0（同梱の tt-lang 1.1.5）、torch 2.11.0、torch-xla 2.9.0、p150a のファームウェア 19.15.0。この README と `docs/ja/discrepancies.md` に記載した制約には、このバージョンに固有のものがある。

## 知見

- trace を用いる前は、ホストが演算を 1 つずつデバイスへディスパッチする時間が、1 ステップの時間を律速していた。ディスパッチのコストは演算 1 つあたり約 70〜360 µs で、A30 の 12〜72 倍。
- ニューロンを要素ごとの演算で記述すると、1 層 1 時刻あたり約 7 演算になる。そのため LIF が演算数の最大の要因だった。最初に行うべきは、ニューロンを 1 層につき 1 カーネルに融合すること。
- trace の導入後は、IR を演算ごとに再実行して得た稼働時間が、実測の時間をよく予測した（9.19 ms と 9.44 ms）。次の施策はこの内訳から選べる。
- デバイス上では、当初は conv よりも並べ替え（`reshape`、`permute`、`to_layout`）のほうが時間を要していた。32 × 32 のタイルの倍数でない形状の並べ替えは、1 回あたり約 1 ms かかる。ニューロンのカーネルを前後の演算と同じ順序で呼び出せば、並べ替えは不要になる。
- スパイクは 0 か 1 で、SEW の加算の結果も小さな整数であるため、どの精度でも正確に表現できる。そのため、残差結合の加算を LIF と融合しても、平均プーリングを 0/1 の行列との積に置換しても、logits はビット単位で一致した。
- 並べ替えを排除すると、残るのは conv と LIF。stem を除く conv の時間は A30（TF32）とほぼ同等で、LIF は DRAM 帯域の上限にほぼ達した。IR の再実行では、活性化を L1 に配置すると稼働時間が約 1.9 ms 短縮した。ただし、tt-xla 1.4.0 にはそれを指定する手段がない。

## 再現手順

環境の構築手順は `docs/ja/setup.md` に記載した。パスは環境変数で表記している。

| 変数 | 内容 |
| --- | --- |
| `$VENV_XLA140` | Python 3.12 の venv。`pjrt-plugin-tt==1.4.0`、torch-xla 2.9.0、SpikingJelly 2.0.0（`--no-deps -e` でインストール） |
| `$UV_PY312_LIB` | uv でインストールした Python 3.12 の `lib`。torch_xla が libpython を見つけるために必要 |
| `$VENV_A30` | torch 2.11.0+cu128 と SpikingJelly 2.0.0 の venv |
| `$DATA` | 結果の出力先 |

```bash
# p150a: まとめて同期で 20 ステップ。batch 8、T = 4、224 × 224、optimization level 1、trace あり
NEURON=ttl LIF_LAYOUT=nhwc4 BN=eval STEM_PAD=32 STEM_FOLD=1 STEM_SPLIT=2 STEM_LIF4=1 INPUT_LAYOUT=nhwc \
LIF_GRID=11,10 LIF_RES=1 AVGPOOL=matmul WAIT=cpu LD_LIBRARY_PATH=$UV_PY312_LIB \
  $VENV_XLA140/bin/python -I tools/xla-sew18.py compile 8 4 20 224 1 trace

# A30 の基準（TF32、cudnn.benchmark）
CUDA_VISIBLE_DEVICES=0 TF32=1 CUDNN_BENCH=1 BN=eval $VENV_A30/bin/python -I tools/a30-sew18.py LIFNode 8 4 224 30
```

どちらの測定でも、ホストのスレッドはカードと同じ NUMA ノードに固定した。p150a の値は 3 回の実行の中央値。

## ファイル構成

| パス | 内容 |
| --- | --- |
| `tools/ttl_lif.py` | tt-lang で記述した、複数時刻の LIF のカーネル（`make_lif`、`make_lif4`、`make_lif2`、`make_lif_add`、`make_lif_nchw`） |
| `tools/xla-sew18.py` | tt-xla で SEW-ResNet18 を p150a 上で実行し、測定する。表の各段階は環境変数で切り替える |
| `tools/a30-sew18.py`、`tools/a30-sew18-profile.py` | 同じモデルを A30 で測定する。層ごとの GPU 時間も出力する |
| `tools/ttnn-replay-ir.py`、`tools/ttnn-replay-chain.py` | tt-xla が出力した TTNN の IR を演算ごとに（または出力を連結し、L1 に配置して）再実行し、デバイスのプロファイラで測定する |
| `tools/breakdown-sew18.py`、`tools/chain-csv.py` | 種類別、層別の内訳と、時間の下限を算出する |
| `tools/ttl-lif-bench.py` | SEW-ResNet18 の形状で、LIF のカーネルのデバイス時間を測定する |
| `tools/xla-op-hostcost.py`、`tools/thread-sampler.py` | 演算 1 つあたりのホスト側のコストを測定する。待機中のスレッドごとの CPU 周波数を記録する |
| `tools/lif-d22-emul.py`、`tools/xla-conv-check.py`、`tools/xla-stem-pad.py`、`tools/xla-bn-eval.py`、`tools/xla-op-backward.py`、`tools/sj_xla.py` | `discrepancies.md` の制約を再現するスクリプトと回避策 |
| `docs/ja/` | 作業記録（日本語） |
| `docs/figures/` | この README の図 |

## 作業記録

`docs/ja/` には次の 3 つの文書がある。

- `speedup-guide.md`: 他の SNN を高速化する際の手順。効果のあった施策と、なかった施策。
- `discrepancies.md`: 前提と実際の相違、ツールの制約と不具合（D1〜D32）。
- `setup.md`: 環境、スクリプトの使用方法、LIF のカーネルの式。

文書とコードのコメント中の S 番号は、非公開の作業記録の番号。

## 次の段階

Blackhole 上で、重み、膜電位、スパイクを Tensix コアの L1 に保持したまま、全コアを 1 時刻に 1 回同期させて SNN を実行する方式の開発を進めている。

## ライセンス

Apache-2.0（`LICENSE`）。スクリプトが用いる SpikingJelly、tt-xla、tt-lang、TT-NN も Apache-2.0 で、このリポジトリには含めていない。

## 引用

`CITATION.cff` を参照。
