# Spiking ResNet on Tenstorrent

[English](README.md)

SpikingJelly 2.0 の SEW-ResNet18 の推論を、Tenstorrent Blackhole p150a で動かして速くした記録です。モデルは tt-xla（torch_xla と PJRT プラグインの `pjrt-plugin-tt`）でコンパイルし、LIF ニューロンだけを tt-lang で書いたカーネルに置き換えました。

- 推論 1 ステップ（T = 4 時刻ぶん）の時間は、222.8 ms から 9.44 ms になりました
- NVIDIA A30（TF32）では、同じ 1 ステップが 7.52 ms です。p150a の時間はその 1.26 倍です

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/step-times-dark.png">
  <img alt="変更ごとの 1 ステップの時間。222.8 ms から 9.44 ms まで下がり、A30（TF32）の 7.52 ms に近づく" src="docs/figures/step-times-light.png" width="800">
</picture>

## 結果

条件: SEW-ResNet18（接続は ADD）、入力 224 × 224、batch 8、T = 4、fp32、BatchNorm は推論モード。

- まとめて同期: k ステップを続けて送り、最後に 1 回だけ待つ。かかった時間を 1 ステップあたりに直した値
- 1 ステップごとに同期: 毎ステップ、logits をホストに読み戻す

| 段階 | まとめて同期 | 1 ステップごとに同期 |
| --- | --- | --- |
| 元の `SimpleLIFNode`（LIF を TTNN の小さな演算で流す） | 222.8 ms | 255.8 ms |
| LIF を tt-lang の 1 カーネルにする（BN の出力の並び NCHW で呼ぶ） | 70.1 ms | 85.7 ms |
| LIF のカーネルを conv の出力の並びで呼ぶ | 31.5 ms | 43.0 ms |
| LIF の前後の reshape を消す。layer4 は `[1, T, R, C]` で渡す | 28.3 ms | 34.6 ms |
| trace を使う | 21.1 ms | 27.2 ms |
| 入力をホストで conv の並びにして渡す | 17.1 ms | 22.7 ms |
| stem の conv を 2 × 2 で折りたたむ | 14.5 ms | 19.6 ms |
| stem の conv を時刻ごとに分け、LIF が直接読む | 11.9 ms | 17.3 ms |
| LIF のカーネルの計算を減らし、110 コアで動かす | 10.1 ms | 15.5 ms |
| 残差の足し算を LIF に取り込む | 9.59 ms | 14.2 ms |
| stem の conv を 16 枚 × 2 に分け、平均プーリングを行列積にする | **9.44 ms** | 15.0 ms |
| （参考）A30、TF32 と `cudnn.benchmark` | 7.52 ms | 7.63 ms |
| （参考）A30、SpikingJelly のベンチマークと同じ厳密な fp32 | 14.75 ms | — |

最後の構成で、1 ステップの時間がどこに使われているかを分けると、次のようになります。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/breakdown-dark.png">
  <img alt="最後の構成の 1 ステップの内訳。p150a は conv 3.71 ms、LIF 2.11 ms、並べ替えなど 1.30 ms、プーリングなど 1.11 ms、stem の conv 0.96 ms。A30 は conv 3.25 ms、LIF 1.54 ms、stem の conv 1.06 ms、BN 0.97 ms、プーリングなど 0.58 ms" src="docs/figures/breakdown-light.png" width="800">
</picture>

p150a の値は、tt-xla が書き出した TTNN の IR を、演算ごとにデバイスで流し直して測った稼働時間の和です（9.19 ms。実際の 1 ステップは 9.44 ms）。流し直しにはコンテナの tt-metal を使いました。tt-xla が使う tt-metal とはバージョンが違うので、目安の値です。A30 の値は torch.profiler で測った GPU のカーネル時間の和です（7.41 ms。実際は 7.52 ms）。p150a では、BN は conv に畳み込まれ、残差の足し算は LIF に取り込まれています。

### 結果を読むときの注意

- 測ったのは速さだけです。重みは初期値のままで、入力は乱数です（SpikingJelly のベンチマークと同じ）。学習済みの重みでの精度は見ていません
- 正しさは同じプロセスで CPU と層ごとに比べて確かめました。数学を変えないはずの変更（並べ替えの削除、分割、融合）では、logits がビット単位で変わらないことを確かめました
- p150a の conv の出力は仮数 10 ビット程度なので、しきい値の近くでスパイクが反転します。LIF をカーネルにした時点で、デバイスの logits と CPU の logits の相関は 0.911 でした（デバイス上の元の `SimpleLIFNode` は 0.908）。CPU だけで層の出力に丸め程度の乱れを入れても、相関は 0.92 前後で頭打ちになります
- 後半の段階では、入力の並べ替え、チャネルのゼロ埋め、折りたたみ、時刻ごとの分割をホストで済ませています（trace の下で約 6.6 ms）。この時間は p150a の値に入っていません
- A30 の値は同じモデルを同じ組み方で動かして測ったものです（LIF は SpikingJelly の `LIFNode` で、A30 では Triton のカーネルで動きます）。SpikingJelly のベンチマーク自体は TF32 を切るので、両方の値を載せました
- バージョン: `pjrt-plugin-tt` 1.4.0（同梱の tt-lang 1.1.5）、torch 2.11.0、torch-xla 2.9.0、p150a のファームウェア 19.15.0。この README と `docs/ja/discrepancies.md` に書いた制約には、このバージョンに限るものがあります

## 分かったこと

- trace を使う前は、ホストが演算を 1 つずつデバイスに送る時間で 1 ステップの時間が決まっていました。送る費用は演算 1 つあたり約 70〜360 µs で、A30 の 12〜72 倍です
- ニューロンを要素ごとの演算で書くと、1 層 1 時刻あたり約 7 演算になります。そのため LIF が演算の最大の出どころでした。最初にやることは、ニューロンを 1 層 1 カーネルにすることです
- trace の後は、IR を演算ごとに流し直した稼働時間が、実際の時間をよく予測しました（9.19 ms と 9.44 ms）。次の手はこの内訳を見て選べます
- デバイスの上では、最初は conv よりも並べ替え（`reshape`、`permute`、`to_layout`）のほうが時間がかかっていました。32 × 32 のタイルの倍数でない形の並べ替えは、1 回で約 1 ms かかります。ニューロンのカーネルを前後の演算と同じ並びで呼ぶと、並べ替えは消えます
- スパイクは 0 か 1 で、SEW の足し算の結果も小さな整数なので、どの精度でも正確に表せます。そのため、残差の足し算を LIF に取り込んでも、平均プーリングを 0/1 の行列との積にしても、logits はビット単位で同じでした
- 並べ替えを消すと、残るのは conv と LIF です。stem を除く conv の時間は A30（TF32）とほぼ同じで、LIF は DRAM の帯域でほぼ頭打ちになりました。IR の流し直しでは、活性化を L1 に置くと稼働が約 1.9 ms 縮みました。ただし、tt-xla 1.4.0 にはそれを指定する手段がありません

## 再現の仕方

環境の作り方は `docs/ja/setup.md` にあります。パスは環境変数で書いています。

| 変数 | 中身 |
| --- | --- |
| `$VENV_XLA140` | Python 3.12 の venv。`pjrt-plugin-tt==1.4.0`、torch-xla 2.9.0、SpikingJelly 2.0.0（`--no-deps -e` で入れる） |
| `$UV_PY312_LIB` | uv が入れた Python 3.12 の `lib`。torch_xla が libpython を見つけるために要る |
| `$VENV_A30` | torch 2.11.0+cu128 と SpikingJelly 2.0.0 の venv |
| `$DATA` | 結果を書き出す場所 |

```bash
# p150a: まとめて同期で 20 ステップ。batch 8、T = 4、224 × 224、optimization level 1、trace あり
NEURON=ttl LIF_LAYOUT=nhwc4 BN=eval STEM_PAD=32 STEM_FOLD=1 STEM_SPLIT=2 STEM_LIF4=1 INPUT_LAYOUT=nhwc \
LIF_GRID=11,10 LIF_RES=1 AVGPOOL=matmul WAIT=cpu LD_LIBRARY_PATH=$UV_PY312_LIB \
  $VENV_XLA140/bin/python -I tools/xla-sew18.py compile 8 4 20 224 1 trace

# A30 の基準（TF32、cudnn.benchmark）
CUDA_VISIBLE_DEVICES=0 TF32=1 CUDNN_BENCH=1 BN=eval $VENV_A30/bin/python -I tools/a30-sew18.py LIFNode 8 4 224 30
```

ホストのスレッドはどちらの計測でも、カードと同じ NUMA ノードに固定しました。p150a の値は 3 回の実行の中央値です。

## ファイルの一覧

| パス | 中身 |
| --- | --- |
| `tools/ttl_lif.py` | tt-lang で書いた複数時刻の LIF のカーネル（`make_lif`、`make_lif4`、`make_lif2`、`make_lif_add`、`make_lif_nchw`） |
| `tools/xla-sew18.py` | tt-xla で SEW-ResNet18 を p150a に流して測る。表の各段階を環境変数で切り替えられる |
| `tools/a30-sew18.py`、`tools/a30-sew18-profile.py` | 同じモデルを A30 で測る。層ごとの GPU の時間も出す |
| `tools/ttnn-replay-ir.py`、`tools/ttnn-replay-chain.py` | tt-xla が書き出した TTNN の IR を演算ごとに（または出力をつないで、L1 に置いて）流し直し、デバイスのプロファイラで測る |
| `tools/breakdown-sew18.py`、`tools/chain-csv.py` | 種類ごと、層ごとの内訳と、時間の下限を出す |
| `tools/ttl-lif-bench.py` | SEW-ResNet18 の形で、LIF のカーネルのデバイス時間を測る |
| `tools/xla-op-hostcost.py`、`tools/thread-sampler.py` | 演算 1 つあたりのホストの費用を測る。待っている間の、スレッドごとの CPU の周波数を記録する |
| `tools/lif-d22-emul.py`、`tools/xla-conv-check.py`、`tools/xla-stem-pad.py`、`tools/xla-bn-eval.py`、`tools/xla-op-backward.py`、`tools/sj_xla.py` | `discrepancies.md` の制約を再現するスクリプトと回避策 |
| `docs/ja/` | 作業の記録（日本語） |
| `docs/figures/` | この README の図 |

## 作業の記録

`docs/ja/` に、次の 3 つの文書があります。

- `speedup-guide.md`: ほかの SNN を速くするときの進め方。効果のあった手と、なかった手
- `discrepancies.md`: 前提と実際が違っていたこと、ツールの制約と不具合（D1〜D32）
- `setup.md`: 環境、スクリプトの使い方、LIF のカーネルの式

文書とコードのコメントにある S 番号は公開していない作業記録の番号です。

## 次の段階

Blackhole の上で、重み、膜電位、スパイクを Tensix コアの L1 に置いたまま、全コアを 1 時刻に 1 回同期させて SNN を動かす方向へ進めています。

## ライセンス

Apache-2.0 です（`LICENSE`）。スクリプトが使う SpikingJelly、tt-xla、tt-lang、TT-NN も Apache-2.0 で、このリポジトリには含めていません。

## 引用

`CITATION.cff` を見てください。
