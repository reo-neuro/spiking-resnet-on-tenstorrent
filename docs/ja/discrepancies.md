# 前提と実際の相違

SNN の検証で、前提と実際が異なっていた事項を記録する。同じ思い込みを繰り返さないためと、TT や SpikingJelly の制約を一か所で参照できるようにするためである。

種類は 2 つある。

- 判断: エージェント（またはユーザー）が確認せずに置いた前提が、実装や実測と異なっていたもの
- 実装: 公式の文書や一般的な期待と、TT や SpikingJelly の実際の動作が異なっていたもの

新しい項目は末尾に D 番号で追加し、番号は振り直さない。解決したら「扱い」を書き換え、項目は削除しない。確定した制約は `setup.md` にも記載する。経緯の詳細は作業記録の S 番号で参照する（作業記録は公開していない）。

## 判断の教訓

- 回避策を勧める前に、TT 側の実装（tt-xla のテスト、公式のベンチマークでの使用方法）を読む（D3）。
- 「動作しない」と結論する前に、失敗したテストが何を実行していたか（forward のみか、backward もか）を確認する（D4）。
- 公式が推奨する実行経路（PyTorch なら `torch.compile(backend="tt")`）を先に試す。別の経路で見つけた問題は、公式の経路で再現するかを確認してから一般化する（D5）。
- ライブラリの使用方法は、手元にあるバージョンの文書とコードで確認する。記憶にある古いバージョンの API で説明しない（D1、D2）。
- 「固定費」は、入力の大きさにも実装方法にもよらず、本質的に削減できないものにだけ使う。batch によらないだけのもの（演算の数に比例するディスパッチのコストなど）を固定費と呼ばない（D16）。
- 移植の作業では、ライブラリの中心となる構成要素（SNN なら LIF）を先にターゲットのカーネルとして実装する計画を立てる。周辺の設定（最適化レベル、trace）の調査から始めない（D16）。
- ローカルの clone にある skill や文書を「公式の現行のもの」として示す前に、upstream の同じ場所に残っているかを確認する（D17）。
- ベンチマークの条件（学習モードか推論モードか、精度、batch）は、ベンチマークのコードの該当行で確認する。比較対象と同じ条件かを、数値を比較する前に確認する（D23）。
- 正確性をネットワークの最終出力の相関だけで判断しない。出力がバイアスに支配されていると、途中の層が誤っていても相関は高くなる。層ごとに CPU と比較する（D24）。
- 正確性の指標が低下したとき、精度（丸め）が原因と決める前に、層ごとに出力の分布（スパイクの発火率、値の種類ごとの割合）を CPU と比較する。丸めはしきい値の近傍でスパイクを散発的に反転させるだけで、分布はほとんど変えない。分布が変わるなら別の誤りがある（D25）。
- SEW-ResNet18 の logits の相関は、どこかに 1e-5 以上の摂動が加わると 0.92 前後まで低下して下げ止まる（推論モードの BN、stem をゼロパディングしたバージョン。S19）。小さな精度の差の大小は測定できず、それより大きく低下したときだけ系統的な誤りを示す。
- 時間を測定するときは、デバイスの完了まで待機しているかを、出力をホストへ転送するバージョンと比較して確認する。tt-xla の `torch_xla.sync(wait=True)` は待機しない。カーネルの帯域はデバイスのプロファイラのカーネルの区間から算出し、呼び出し 1 回の壁時計時間から算出しない（D21）。
- ホストとデバイスの往復を含む時間が、同じ条件でプロセスごとに二峰に分かれたら、デバイスを疑う前にホストの CPU の実際の周波数を確認する（D31）。

## 一覧

| D | 種類 | 内容 | 扱い | 経緯 |
| --- | --- | --- | --- | --- |
| D1 | 判断 | SpikingJelly の GPU の比較対象を cupy カーネルとした | 2.0 では Triton かネイティブ CUDA。訂正済み | Web 検索 |
| D2 | 判断 | multi-step に `backend='torch'` を指定すれば純 PyTorch で実行されるとした | 2.0 に `backend` 引数はなく、xla では `SimpleLIFNode` を用いる | S2 |
| D3 | 判断 | 推論の BN を手作業で Conv に統合することを勧めた | TT のコンパイラ自身が統合する。BN は eval のまま渡す | S3 |
| D4 | 判断 | 「eval モードの BN はコンパイルできない」とした | backward も実行するテストだったため。forward は動作する | S3 |
| D5 | 判断 | lazy tensor の経路だけで検証を進めた | 公式は `torch.compile(backend="tt")`。学習時の誤りは公式の経路で未確認 | S3 |
| D6 | 実装 | `LIFNode` などの登録済みのニューロンが xla で動作しない | `SimpleLIFNode` を用いる | S2 |
| D7 | 実装 | Conv の直後の `nn.BatchNorm2d`（学習モード）の backward の結果が誤る | `ManualBN2d` で回避。原因は未特定 | S2 |
| D8 | 実装 | Conv の後の `nn.GroupNorm` の入力勾配もずれる | 使用していない。原因は未特定 | S2 |
| D9 | 実装 | optimization_level 1・2 で、Conv を含むグラフがコンパイルできない | 1.5.0.dev20260831 の退行。1.4.0 なら level 1 は通る | S2、S3、S8 |
| D10 | 実装 | A30 で CLIF の Triton カーネルの境界値のテストが 1 件失敗する | LIF には関係しないので保留 | S1 |
| D11 | 判断 | `SimpleLIFNode` は single-step の実装なので、T のループを外側で回す必要があるとした | `multi_step_forward` があり、multi-step のまま使用できる | S4 |
| D12 | 実装 | SEW-ResNet18 が batch 12 以上で実行時にエラーになる | batch 8 以下で測定する。1.5.0 は batch 16 で、1.4.0 は batch 12 と 16 で確認した。tt-mlir #8559（未解決）と同じ症状と見ている | S4、S9、S10 |
| D13 | 判断 | TT の index に、使用中のバージョンより新しい pjrt-plugin-tt があるとした | 公開されている最新のバージョンは使用中のもの（08-31）。新しい tt-mlir を用いるにはソースからのビルドが必要 | S5 |
| D14 | 判断 | SEW-ResNet18 の固定のコスト（約 315 ms）は、演算を 1 つずつホストからディスパッチする時間だとした | デバイス上の時間が batch 8 で 243 ms あり、batch に比例する | S6 |
| D15 | 実装 | trace が optimization_level 0 では Conv を含むとコンパイルできない | 小さいモデル（Conv＋BN＋ReLU）では 1.4.0 の level 1・2 でも abort する。ただし SEW-ResNet18（level 1、推論モードの BN）は trace で実行できる（S23） | S6、S8、S23 |
| D16 | 判断 | 演算のディスパッチのコストを「固定費」と呼び、LIF のカーネル化を後回しにしてプラグインの設定の調査を続けた | ディスパッチのコストは演算の数に比例し、LIF の融合で削減するのが本筋。LIF のカーネル化を中心の作業にする | S11、S12 |
| D17 | 判断 | 手元の `~/tt-metal` の `.agents/skills` を、現行の公式の skill として示した | upstream では 09-01 に削除され、`tenstorrent/skills` に移行した。再導入済み | S13 |
| D18 | 実装 | 同じ tt-lang のカーネルを 1 つのグラフで 2 回以上呼び出すと、2 回目の resolve が失敗する | tt-lang のカーネルごとのキャッシュのため、tt-xla のフックが機能しない。呼び出すたびにカーネルを再生成して回避 | S14 |
| D19 | 実装 | tt-lang の仕様（手元の clone、09-30）の記述方法が、tt-xla 1.4.0 の wheel の tt-lang 1.1.5 では通らない | `>=`、`where`、`mask`、`clamp` がなく、fp32 の CB の制約もある。`sign(relu(floor(h)))` などで構成した | S14 |
| D20 | 実装 | 同じ条件の 1 ステップの時間は一定だとした | デバイスに 0.3〜0.4 秒続けて送信すると最大 2 倍遅くなり、3 秒休止すると元に戻る。出力をホストへ転送して待機すると解消するので、ホストがデバイスに先行する分と見ている（S16、D21） | S15、S16 |
| D21 | 判断 | `torch_xla.sync(wait=True)` までの時間を、デバイスの処理の時間として扱った | デバイスの完了を待機しない。LIF の 1 回は 30.5 µs と計測されたが、ホストへ転送して待機すると 1,315 µs。S4〜S15 の 1 ステップの時間は、ホストのディスパッチの時間か、デバイスに追いつかれた後の時間 | S16 |
| D22 | 実装 | tt-lang のカーネルの fp32 の計算は fp32 の精度で行われる | 2 つの CB を加算すると FPU で仮数約 10 ビットに丸められる。SFPU の経路なら正確だが、それには CB ごとの `UnpackToDestFp32` が必要で、tt-xla 1.4.0 の tt-lang 統合はそれを渡さない。tt-xla の経路では CPU とビット単位で一致させられない。読み込み時の丸めは切り捨てで、SEW-ResNet18 への影響は conv の丸めより 2 桁小さい | S16、S17、S19 |
| D23 | 判断 | SpikingJelly のベンチマークは推論でも `model.eval()` を呼び出さないので、BN は学習モードだとした（S4） | 推論のフェーズでは `model.train(args.phase == "training")` で推論モード。A30 は推論モード、p150a は学習モードの BN で比較していた。学習モードの BN と前後の並べ替えが、LIF なしの 123 ms のうち約 94 ms を占めていた | S4、S18 |
| D24 | 実装 | tt-xla 1.4.0 の optimization_level 1 は、conv を正しく計算する | 入力が 3、4、8、16 チャネルの stem の conv（7×7、stride 2）だけ、CPU との相関が 0.21〜0.29 になる。32 チャネルにゼロパディングすると正しい。level 1 の SEW-ResNet18 の出力は S10〜S17 のあいだ誤っていた | S18 |
| D25 | 実装 | tt-lang の演算の出力先を `torch.zeros_like` で生成すれば、呼び出しごとに別のバッファになる | tt-mlir は同じ形状の 0 の定数を 1 つに共通化し、カーネルはそこに直接書き込むので、同じ形状の LIF がすべて 1 つのバッファを共有する。2 つの LIF の出力を加算するブロックの結果が誤り、logits の相関が 0.908 ではなく 0.731 になった。LIF ごとに異なる値で埋めた出力先を渡して回避 | S19 |
| D26 | 判断 | A30 と p150a の SEW-ResNet18 を、同じ fp32 の精度で比較しているとした | A30 の基準（SpikingJelly のベンチマーク）は TF32 を無効にした厳密な fp32 の conv、p150a の conv は仮数 10 ビット相当。p150a に有利な条件の比較になっている。TF32 を有効にすると A30 は 14.75 ms から 7.5 ms | S19、S20 |
| D27 | 判断 | LIF のカーネルは、1 回に転送するブロックが小さいので読み書きが律速だとした（S22 での推定） | SFPU の計算が約 7 割だった。tt-lang は `store` ごとに区間を分割して初期化を繰り返し、2 つの `store` に共通の部分式も別々に計算する。しきい値の判定を `lt` と `fill` で実装して計算を削減し、ブロックを拡大して、20 層 2.81 → 1.92 ms | S25 |
| D28 | 実装 | tt-lang のカーネルの `grid="auto"` は、tt-xla の経路でも実機の計算コアの格子（p150a は 11×10）になる | tt-xla 1.4.0 の tt-lang 統合は、仮の 8×8 の Wormhole の device でカーネルを構築するので、64 コアになる（`arch` も `wormhole_b0`）。S24、S25 のモデルの LIF は 64 コアで実行されていた。grid を明示すれば 110 コアで実行される | S26 |
| D29 | 実装 | コンテナで正しく動作した tt-lang のカーネルは、tt-xla の経路でも動作する | stem の LIF が bf16 のスパイクを書き込むバージョン（`make_lif4(bf16_out=True)`）は、コンテナでは正しく動作したが、SEW-ResNet18 の中ではデバイスがハングし、`tt-smi -r` が必要になった。原因は分かっていない | S26 |
| D30 | 実装 | プラグインに tt-mlir の `override-conv2d-config` があるので、tt-xla から conv の設定や L1 への配置を指定できるかもしれない | 1.4.0 が受け付けるのは決まったキーのみで、未知のキーは警告なしに破棄する。level 0、1 は活性化をすべて DRAM に配置する。L1 に配置し、conv の double buffer を用いれば、stem を除く conv と LIF で約 1.3 ms 短縮する（コンテナでの単体の測定）が、tt-xla の経路では使用できない | S29 |
| D31 | 判断 | 1 ステップごとに同期するときの追加の時間（約 5 ms）は、tt-xla の実行時のコスト（trace の起動、ホストへの転送、ディスパッチ）だとした | 大半はホストの CPU の省電力機能による。待機のたびにスレッドがスリープし、CPU は 800 MHz のまま動作する。SMT の兄弟の論理 CPU でビジーループを実行すると 14.2 → 11.0 ms。対処するかはユーザーが決める | S24、S31 |
| D32 | 実装 | tt-xla で、入力をタイルレイアウトで渡せるかもしれない（ホストでタイルレイアウトに変換するかは比較条件の判断） | tt-xla 1.4.0 の trace の入力は常にホストの行優先レイアウトとして扱われ、デバイス上のタイルレイアウトのテンソルを渡しても trace の中でタイルレイアウトに変換する。手段がない | S24、S33 |

## 各項目

### D1. GPU の比較対象は cupy カーネルとした（判断）

- 前提: SpikingJelly の CUDA の高速化は cupy のバックエンドで行う（エージェントの記憶。0.0.0.0.14 までの構成）。
- 実際: 2.0.0（2026-10-17）のバックエンドは torch、Triton、ネイティブ CUDA。README に「no CuPy dependency required」とある（Web 検索で確認した）。
- 影響: 比較の基準を Triton とネイティブ CUDA に改めた。
- 扱い: 訂正済み

### D2. `backend='torch'` で純 PyTorch の経路になるとした（判断）

- 前提: multi-step のニューロンに `backend='torch'` を指定すれば、カスタムカーネルを経由しない（エージェントの記憶。古いバージョンの API）。
- 実際: 2.0.0 の `LIFNode` に `backend` 引数はなく、バックエンドはテンソルのデバイスから自動で選択される。CPU でも xla でも、`torch.library` のカスタム演算子を経由する（D6）。
- 影響: 最初の実行で `LIFNode` がエラーで停止した。純 PyTorch の `SimpleLIFNode` に切り替えた。
- 扱い: 訂正済み

### D3. 推論の BN を手作業で Conv に統合することを勧めた（判断）

- 前提: eval モードの BN はコンパイルできないので（D4）、推論では BN を Conv に統合するのがよい（一般的な推論の最適化からの推測。TT 側は確認していなかった）。
- 実際: tt-xla の `tests/torch/graphs/fusion_patterns/test_batch_norm_decomposition.py` によると、Conv の後の eval の BN はコンパイラが `alpha*x + beta` に分解し、`Conv2dWithMultiply` の融合で Conv の重みに統合する（optimization_level 1 以上）。公式の ResNet-50 のベンチマークも BN を残したまま実行している。
- 影響: ユーザーに指摘されて確認した。手作業で統合すると、コンパイラの処理と重複する。
- 扱い: BN は eval のまま渡す。ただし融合が有効になる 1・2 は現在は使用できない（D9）。

### D4. eval モードの BN はコンパイルできないとした（判断）

- 前提: `tools/xla-op-backward.py` の `bn_eval` が `The size of mean should be compatible with feature count` で失敗したので、eval の BN は TT で使用できない。
- 実際: そのテストは eval の BN でも backward を実行していた。forward のみの `tools/xla-bn-eval.py` では、lazy でも `torch.compile` でも動作する（optimization_level 0、相関 0.99991）。
- 影響: D3 の誤った推奨の根拠になった。
- 扱い: 訂正済み。eval の BN の backward は現在も失敗するが、使用する場面はない。

### D5. lazy tensor の経路だけで検証を進めた（判断）

- 前提: テンソルを xla デバイスに配置して eager に記述すれば、torch_xla の標準的な使用方法になる（blacksmith の MNIST がこの形式だった）。
- 実際: tt-xla の文書と公式のベンチマークは `torch.compile(..., backend="tt")` を用いる。
- 影響: D7、D8、D9 は lazy の経路で見つけたもの。D9 は `torch.compile` の経路でも再現した。D7 と D8 は `torch.compile` の経路ではまだ確認していない。
- 扱い: 以降は `torch.compile(backend="tt")` を主に用いる。

### D6. 登録済みのニューロンが xla で動作しない（実装）

- 期待: SpikingJelly 2.0 の README では、torch の参照実装が CPU と、拡張がないときの経路になっている。純 PyTorch なら xla でも動作すると考えた。
- 実際: 2.0.0 のニューロンは `torch.library` のカスタム演算子として、CPU と CUDA のディスパッチキーにだけ実装が登録されている（`ops/dispatch.py`）。xla では `RuntimeError: SpikingJelly neurons support CPU and NVIDIA CUDA, not xla:0`。`functional.neuron_implementation` も `ValueError: Unsupported neuron device xla:0`。
- 影響: SEW-ResNet18 などのモデルは、ニューロンを `SimpleLIFNode` に置換しないと動作しない。
- 扱い: `SimpleLIFNode` を用いる。参照実装を XLA のキーにも登録するパッチを当てる案もある（未着手）。

### D7. Conv の直後の学習モードの BN の backward の結果が誤る（実装）

- 期待: 単独で正しい演算子は、組み合わせても正しい。
- 実際: Conv と BN はそれぞれ単独なら CPU と一致する（相関 0.999 以上）。Conv → BN の順につなぐと、Conv の重みの勾配の最大値が CPU の 535 に対して 51,400 になる（相関 0.11）。精度のオプションを変更しても結果は同じ。
- 影響: そのままでは学習の loss が CPU の値から乖離していく。
- 扱い: `tools/sj_xla.py` の `ManualBN2d` で回避。原因は未特定（Conv の出力の NHWC の順序と、BN の backward の軸の取り違えを疑っている）。`torch.compile` の経路では未確認。

### D8. Conv の後の GroupNorm の入力勾配がずれる（実装）

- 期待: D7 の回避策として GroupNorm が使用できる。
- 実際: Conv → `nn.GroupNorm(4, 16)` で、入力の勾配の相関が 0.90、Conv の重みの勾配の相関が 0.58。
- 影響: 正規化の層の変換処理に共通の問題がある可能性がある。
- 扱い: 使用していない。原因は未特定。

### D9. optimization_level 1・2 で Conv を含むグラフがコンパイルできない（実装）

- 期待: 公式の ResNet-50 のベンチマークは optimization_level 2 で動作している。
- 実際: pjrt-plugin-tt `1.5.0.dev20260831000501` で、1 は `ttnn.permute failed validation`（`vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)`）、2 はコンパイラ（`libTTMLIRCompiler.so`）の中で segfault。lazy でも `torch.compile` でも同じ。
- 影響: BN の融合などの最適化が使用できず、p150a の速度は公式の条件より不利になる。
- 扱い: 0 で測定し、比較には但し書きを付ける。新しいプラグインで修正されているかを確認する（作業記録の「次にやること」）。
- 追記（S6）: Conv 1 個（16→32、3×3）でも、bf16 でも、公式の例 `examples/pytorch/mnist_performant.py`（level 2、trace）でも発生する。モデルの構成の問題ではない。segfault のスタックは `OpModel<PermuteOp>::getOpConstraints` → `ttnn::permute` → `ProgramImpl::update_kernel_groups` で、1 の `_M_range_check` と同じく、コンパイラが制約を問い合わせる経路の中で発生している。このプラグインは sfpi 7.69.0 を想定しているが、ホストの `/opt/tenstorrent/sfpi` は 7.83.0（2026-10-22 に 7.72.0 から更新された）。
- 追記（S7）: 7.69.0 を venv の中（`pjrt_plugin_tt/tt-metal/runtime/sfpi`）に配置しても、level 1 の Conv 1 個は同じ `_M_range_check` で、`mnist_performant.py` は同じ `update_kernel_groups` で失敗した。level 1 のエラーはカーネルのビルドより前に発生している。sfpi のバージョンは原因ではない。
- 追記（S8）: 同じマシンで、プラグイン 1.4.0（`~/tt-xla-venv`、07-23 ビルド）なら level 1 の Conv（単独でも、eval の BN 付きでも）が通る。blacksmith の venv（1.5.0.dev20260831000501、`.so` は `venv-xla` と同一）では同じく失敗する。**D9 はこのマシンの問題ではなく 1.5.0.dev20260831000501 での退行**である。1.4.0 でも level 2 は、Conv＋BN 1 組のコンパイルが 28 分経過しても終了しなかった（CPU 100%、mock device で探索中。SIGINT では停止せず、SIGTERM で停止させた。デバイスは正常のまま）。

### D10. A30 で CLIF の境界値のテストが失敗する（実装）

- 期待: リリースの検証を通った 2.0.0 なので、テストはすべて通る。
- 実際: `test_clif_scalar_division_matches_torch_at_threshold[1.03]` が失敗する。しきい値ちょうどの入力で、Triton の実装と torch の実装のスパイクが 3 要素中 2 つ一致しない。上流のリリースの検証（PR #778）で GPU として挙がっているのは RTX 4090 で、A30（sm_80）で検証されているかは確認していない。
- 影響: LIF には関係しない。
- 扱い: 保留。CLIF を用いるときに調査する。

### D11. `SimpleLIFNode` は single-step の実装とした（判断）

- 前提: S2 の手順2のスクリプトで 1 ステップずつ呼び出していたので、`SimpleLIFNode` は single-step 専用だと思い込んだ（コードを読んでいなかった）。
- 実際: `SimpleBaseNode.multi_step_forward` が `torch.stack([self.single_step_forward(x) for x in x_seq])` で T のループを回す。`step_mode="m"` で `[T, N, ...]` を受け取れる。
- 影響: S3 で「T のループを Python で回してグラフを展開させる」という不要な案を出していた。
- 扱い: 訂正済み。SEW-ResNet18 は multi-step のまま構成した。Conv は T×N をまとめたバッチで計算され、A30 と同じ形状になる。

### D12. SEW-ResNet18 が batch 12 以上でエラーになる（実装）

- 期待: batch 8 で動作するなら、メモリが足りる範囲で batch を増やせる。
- 実際: batch 16（T×N = 64）と 32 で、実行時に `TT_THROW: Invalid subtile broadcast type`（`binary_ng_device_operation.cpp:224`、`ttnn::add`）。コンパイルは通る。デバイスは正常のまま。
- 追記（S10）: プラグイン 1.4.0 でも発生し、batch 12（T×N = 48）でもエラーになる（`binary_ng_device_operation.cpp:225`）。1.5.0 では batch 12 を試していない。
- 影響: A30 の batch 32（51.1 ms）と同じ条件では比較できない。batch 8 で比較した。
- 扱い: どの加算か（SEW の ADD、BN、LIF の更新）は特定していない。新しいプラグインで修正されているかを確認する。

### D13. TT の index に新しい pjrt-plugin-tt があるとした（判断）

- 前提: nightly の wheel は毎日公開されているので、2026-08-30 ビルドの使用中のバージョンより新しいものが TT の index にある（S4 の「次にやること」。index を確認していなかった）。
- 実際: 2026-10-27 に `pip index versions pjrt-plugin-tt --pre --index-url https://pypi.eng.aws.tenstorrent.com/` を実行すると、最新は `1.5.0.dev20260831000501`（使用中のバージョン）。GitHub の tt-xla のリリースも 08-31 が最後。pypi.org には存在しない。tt-xla の main は 10-14 まで更新が続いており、`third_party/CMakeLists.txt` の `TT_MLIR_VERSION` は `c282803f`（tt-mlir の PR #9279。while 対応の #9187 より後）。
- 影響: 「新しいバージョンを別の venv に入れる」手順はそのままでは進められない。
- 扱い: 新しいバージョンを試すなら、tt-xla をソースからビルドする（`docs/source/getting_started.md` の手順。tt-mlir のツールチェーンもビルドする）。進めるかはユーザーが決める。

### D14. 固定のコストはホストのディスパッチだとした（判断）

- 前提: batch 2 と 8 の壁時計時間（329 ms と 370 ms）の差が小さいので、時間の大半はバッチによらない固定のコストで、その内容は演算を 1 つずつホストからディスパッチする時間である（S4。壁時計時間の 2 点だけからの推測）。
- 実際: デバイスのプロファイラで測定すると、1 回の実行は 1,708 個のプログラムから成り、デバイス上のカーネル時間の合計は batch 2 で 70 ms、batch 8 で 243 ms（batch にほぼ比例し、1 サンプルあたり約 29 ms）。batch 8 の壁時計時間の 3 分の 2 はデバイス上の計算。batch 2 ではデバイスの時間は壁時計時間の 2 割ほどで、残りはホスト側（ディスパッチなど）と見られる。
- 影響: 「trace でディスパッチを排除すれば速くなる」の効果は小さい。batch 8 以上ではデバイス上の計算が律速で、1 サンプルあたり約 29 ms は A30（batch 32 で 1.6 ms）の約 18 倍。
- 扱い: 訂正済み。2 点の壁時計時間から内訳を推測しない。デバイスの時間を測定してから主張する。

### D15. trace が optimization_level 0 で Conv を含むとコンパイルできない（実装）

- 期待: trace（`enable_trace`、`TT_RUNTIME_TRACE_REGION_SIZE`）は optimization_level によらず使用できる（tt-xla の `docs/source/performance.md` に level の条件は記載されていない）。
- 実際: level 0 で、Linear→ReLU→Linear は trace で動作する。Conv→ReLU（fp32 でも bf16 でも）はコンパイルで `error: Memory layout is required for non-SystemMemory buffer type.` を出力して abort する。SEW-ResNet18 は `Device-resident input must be on device, but found on system memory` で停止する。公式のテストで trace を用いるものは、いずれも level 1 以上だった。level 1 は D9 で Conv が通らない。
- 影響: Conv を含むモデルで trace を用いる手段がない。
- 扱い: 新しいプラグインで確認する。デバイスは毎回正常（`fw_id=0`）。
- 追記（S8）: 1.4.0 でも、level 1 の trace と level 2 の trace（bf16）で、Conv＋BN は同じエラーで abort する。バージョンによらない。
- 追記（S23）: 1.4.0 の level 1 で、SEW-ResNet18（推論モードの BN、stem を 32 チャネルにゼロパディングしたバージョン、tt-lang の LIF）は trace で動作した（1 ステップごとに同期 22.7 ms、まとめて同期 17.1 ms）。公式の push テストにも、ResNet-50 を level 1 の trace で実行するもの（`tests/torch/models/resnet/test_resnet.py` の `test_torch_resnet_inference_trace`）がある。小さいモデルとの違い（Conv の bias、入力が直接 Conv に入ることなど）は切り分けていない。「Conv を含むと trace は使用できない」は誤りだった。

### D16. ディスパッチのコストを「固定費」と呼び、LIF のカーネル化を後回しにした（判断）

- 前提: batch を変えても時間が変わらない部分を「固定費」と呼んだ。また、LIF を ttnn か tt-metal で再実装するかは、torch_xla の参照実装の結果を確認してから決めればよいとした（`setup.md` の冒頭の方針）。
- 実際: その部分は演算をデバイスに 1 つずつディスパッチするコストで、演算の数に比例する（1 個あたり約 150〜170 µs、S11）。最大の要因は、LIF を小さな要素ごとの演算に分割して実装していること。A30 でも、LIF を Triton の 1 カーネルで実装した `LIFNode` は `SimpleLIFNode` より約 6 割速い（S12）。SNN のライブラリを移植するなら、LIF のカーネル化が中心の作業になることは、着手前から分かることだった（ユーザーの指摘）。
- 影響: S5〜S12 の多くを、プラグインのバージョン、最適化レベル、trace の調査に費やした。そこで判明したこと（D9 の退行、1.4.0 の level 1、ディスパッチのコストの大きさ）は、LIF のカーネル化の前提として活用できる。
- 扱い: 用語を修正した。「次にやること」の 1 番を LIF の融合カーネルの実装にした。

### D17. upstream から削除された skill を、現行の公式の skill として示した（判断）

- 前提: 手元の `~/tt-metal` の `.agents/skills`（graph-fusing、optimize、tt-lang、tt-device-usage など）を、Tenstorrent の現行の公式の skill として示した（S13 の前半）。
- 実際: upstream の tt-metal からは 2026-09-01 に削除されていた（PR #55086。専用のリポジトリへ移行するため）。現在は `tenstorrent/skills` にプラグインとしてまとめられている。tt-lang の skill は移行しておらず、後継は tt-lang のリポジトリの `claude-slash-commands/` だった。ユーザーが upstream から取り込めないかと尋ねて判明した。
- 影響: 古いバージョンを読んで手段を比較した。graph-fusing の本体は upstream と同じで、比較の結論は変わらない。
- 扱い: upstream から再導入した（`setup.md` の「公式の skill と slash command」）。教訓: ローカルの clone にある skill や文書を「公式の現行のもの」として示す前に、upstream の同じ場所に残っているかを確認する。

### D18. 同じ tt-lang のカーネルを 2 回以上呼び出すと resolve が失敗する（実装）

- 期待: `@tt_torch.tt_lang_operation` でラップしたカーネルは、グラフの中で何回呼び出してもよい。
- 実際: 2 回目の resolve で `TTLangError: tt-lang compile did not produce a CompiledTTNNKernel`。tt-lang（`ttl_api.py`）がカーネルのオブジェクトごとに、形状をキーにしてコンパイル結果をキャッシュしている。キャッシュにヒットすると `_compile_kernel` が呼び出されず、tt-xla（`tt_torch/tt_lang.py` の `_drive_ttl_compile`）のフックは何も捕捉しない。tt-xla の文書は「tt-lang は自分でキャッシュすべき」とだけ記載しており、この組み合わせは想定していないように見える。
- 影響: SEW-ResNet18 のように、同じ形状の LIF が何層もあるモデルでエラーになる。
- 扱い: `@tt_lang_operation` の中で、呼び出すたびに `make_lif(T)` でカーネルを再生成して回避（ラップした関数の本体はグラフのコンパイル時にしか呼び出されない）。tt-xla の新しいバージョンで修正されているかは確認していない。

### D19. tt-lang の仕様の記述方法が、wheel の tt-lang 1.1.5 では通らない（実装）

- 期待: 手元の tt-lang の clone（09-30）の仕様と `/ttl-import` の手順どおりに記述すれば、tt-xla 1.4.0 の wheel の tt-lang（1.1.5.dev20260704）でも通る。
- 実際: 次の点が異なっていた。`>=` がない（`<`、`>`、`==`、`!=` はある）。`ttl.block.where`、`mask`、`clamp` がない（コンテナの 1.1.10 でもシミュレータにしかない）。fill は `ttl.math.fill` ではなく `ttl.block.fill`。カーネルの中で条件式（`a if c else b`）を使用できない。キャプチャした `torch.dtype` を符号化できない（1.1.10）。fp32 では 1 つの CB を FPU の演算（add、sub）と SFPU の演算（mul、floor、スカラーとの乗算など）の両方で読み出せない。
- 影響: LIF の式を再構成する必要があった（S14）。
- 扱い: スパイクを `sign(relu(floor(h)))`、リセットを `v = hb · (1 − s)` にし、h を SFPU 用と FPU 用の 2 つの CB に書き込んだ。tt-lang で記述するときは、仕様ではなく、使用するバージョンの `ttl/_generated_elementwise.py` と `ttl/operators.py` で使用できる演算を確認する。
- 追記（S25）: fill は、モジュールの先頭で `fill = ttl.block.fill if hasattr(ttl, "block") else ttl.math.fill` と別名を定義すれば、1.1.5 と 1.1.10 の両方で使用できる（1.1.5 には `ttl.block` がなく、1.1.10 は `ttl.math.fill` を受け付けない）。型は、カーネルの中に `dtype=torch.float32` と直接記述すれば通る（キャプチャした変数の `torch.dtype` は符号化できないが、モジュールの属性として記述したものは通る）。1.1.5 でも、スカラーとの演算は乗算のみ（加算と減算は 2 つのブロックの間のみ）。
- 追記（S26）: 1 つの計算の区間に fp32 と bf16 のタイルが混在すると、どちらのバージョンもエラーになる（`mixed f32 and non-f32 tile arguments`）。例外は fp32 の入力がすべて直接 `typecast` に渡るとき。1.1.10 では fp32 → fp32 の `typecast` を挟めば通ったが、1.1.5 ではそれが恒等の変換として先に除去され、通らない。fp32 の CB に書き込んでから、別の区間で fp32 → bf16 の `typecast` を適用すれば両方で通る（`make_lif4(bf16_out=True)`）。

### D20. 同じ条件でも 1 ステップの時間が一定にならない（実装）

- 期待: 同じモデル、同じ入力を連続して実行すれば、2 ステップ目以降の時間はほぼ一定になる（S4〜S14 は中央値をそのまま用いた）。
- 実際: SEW-ResNet18（batch 8、level 1）は、休止した直後の 2〜3 ステップが 52〜88 ms で、連続して実行すると 110 ms（LIF なし）または 166 ms（tt-lang の LIF）になる。デバイスが 3 秒休止すると元に戻る。batch 2 では 76 ms のまま 87 ステップ続き、その後 43.5 ms に短縮した。AICLK、電力、温度、Doppler の NOP、ホストの CPU の周波数、NUMA、ほかのプロセスでは説明できない（S15）。
- 影響: 速度の比較が、休止直後の値か連続実行時の値かで最大 2 倍変わる。
- 扱い: 原因は分かっていない（調査には perf か turbostat が必要）。追っても得るものが少ないとユーザーが判断し、調査を中止した（2026-10-27）。休止直後の値と連続実行時の値を分けて報告する。計測は `taskset -c 60-71,156-167` で NUMA 5 に固定する（README の規約。この現象には効果がなかった）。
- 追記（S16）: 毎ステップ出力をホストへ転送して待機すると（`WAIT=cpu`）、SEW-ResNet18 の 2 つの水準は消え、2 ステップ目から一定になった（tt-lang の LIF で 180〜186 ms、LIF なしで 122〜128 ms）。`sync(wait=True)` はデバイスの完了を待機しないので（D21）、休止直後の水準はホストがディスパッチする速度、連続実行時の水準はホストがデバイスに追いつかれた後の速度だと見ている（推測。ホストが先行できる量に上限があり、デバイスが 3 秒休止するとその分が空く）。batch 2 で 87 ステップ目から速くなったことと、コンテナの ttnn の試験（1 回ごとに同期）の 2 つの水準は、これでは説明できない。今後の SEW-ResNet18 の計測は `WAIT=cpu` で行う。

### D21. `torch_xla.sync(wait=True)` までの時間をデバイスの時間として扱った（判断）

- 期待: `torch_xla.sync(wait=True)` は、ディスパッチしたグラフがデバイスで完了するまで待機する。S4〜S15 は、`m(x)` から `sync(wait=True)` までを 1 ステップの時間とし、S14 は K=1 の同じ時間（2.05 ms）からカーネルの帯域（約 100 GB/s）を算出した。
- 実際: tt-xla 1.4.0 で `[4·6272, 1024]` の tt-lang の LIF を K 回連結すると、`sync(wait=True)` だけでは 1 回 30.5 µs で、デバイスのカーネル時間（コンテナのプロファイラで 737 µs）より短い。最後の出力を `.cpu()` でホストへ転送すると 1 回 1,315 µs（S16）。
- 影響: S4〜S15 の 1 ステップの時間は、先行するホストにデバイスが追いつくまではホストのディスパッチの時間で、追いついた後はデバイスの時間に近い（D20）。S14 のカーネルの帯域（約 100 GB/s）は誤りで、コンテナのプロファイラでは 279 GB/s。S6 のデバイスのプロファイラの値は、この影響を受けない。
- 扱い: `xla-ttl-lif.py` と `xla-sew18.py` に `WAIT=cpu`（毎回出力をホストへ転送する）を追加し、以後の計測はこれで行う。カーネルの時間は `tools/ttl-lif-bench.py`（デバイスのプロファイラ）で測定する。

### D22. tt-lang の fp32 の計算は fp32 の精度で行われると思った（実装）

- 期待: fp32 のテンソルを受け取る tt-lang のカーネルは、加減乗除を fp32 で計算する。
- 実際: 2 つの CB を直接加算する（`(v + x) * 0.5` など）と FPU で計算され、fp32 の入力が仮数約 10 ビットに丸められる（`x + x` で正確に一致した要素が 0、相対誤差は最大 9.4e-4。S16）。SFPU を経由する記述（`v + (x * 0.5 - v * 0.5)` など）なら、ttnn で直接実行すると CPU とビット単位で一致する（コンテナの 1.1.10。SEW-ResNet18 の LIF の 5 つの形状で不一致 0。S17）。これは SFPU が直接読み出す CB に tt-lang が `UnpackToDestFp32` を設定するため（wheel の 1.1.5 にもある。`ttl/ttl_api.py` の `_set_unpack_to_dest_fp32`）。しかし tt-xla 1.4.0 の tt-lang 統合（`tt_torch/tt_lang.py` の `_serialize_kernel_config`）は計算カーネルの設定のうち `math_fidelity`、`fp32_dest_acc_en`、`dst_full_sync_en`、`bfp8_pack_precise`、`math_approx_mode` しか渡さず、tt-xla の経路では fp32 の CB の読み込みがすべて丸められる。手元の tt-xla の clone（`b360078`、10-02）も同じ。
- 影響: tt-xla の経路の LIF は、しきい値のごく近傍で CPU と一致しない（`[4·6272, 1024]` の形状で 0.005〜0.007%）。カーネルの記述では修正できない。読み込み時の丸めは切り捨て（round to nearest ではない）で、デバイスの BN の出力に、CPU で同じ切り捨てを施した LIF を適用すると、デバイスの出力とビット単位で一致した（S19）。ただし conv の出力（LIF の入力）がすでに仮数 10 ビットに丸められているので、SEW-ResNet18 の最初の LIF で D22 が加える反転は 46 個で、conv の丸めによる CPU との差（約 17,200 個）より 2 桁少ない。CPU で D22 だけを模擬したネットワークの logits の相関は 0.922 で、ほかの小さな摂動を加えたときと同じ水準（S19）。S18 で「logits の相関が低いのは D22 のため」と推測したのは誤りで、原因は D25 だった。
- 扱い: `make_lif_nchw` は SFPU のみで計算する記述にした（ttnn で直接実行すれば正確で、tt-xla が設定を渡すようになれば正確になる）。tt-xla の新しいバージョンで修正されているかは確認していない。GitHub への報告はユーザーが決める。

### D23. ベンチマークの推論のフェーズで BN は学習モードだとした（判断）

- 期待（S4 で置いた前提）: SpikingJelly の `benchmark/benchmark_snn_single_gpu.py` は推論でも `model.eval()` を呼び出さないので、BN は学習モード（batch の統計で正規化し、running stats を更新する）のまま。p150a でも学習モードにすれば、A30 と条件が揃う。
- 実際: 推論のフェーズでは `model.train(args.phase == "training")`（手元の 2.0.0 の 537 行目。08-10 の `3b12b104` から）で、モデル全体が推論モードになる。A30 の基準（S4 の batch 8 で 14.8 ms、batch 32 で 51.1 ms）は推論モードの BN で測定され、p150a は S4〜S17 のあいだ学習モードの BN で測定していた。p150a では、学習モードの BN（NCHW で、パディング付きのタイルで計算し、前後に `permute` と `reshape` が挿入され、running stats の更新が 1 層あたり 6 個の小さな演算に分割される）と前後の並べ替えが、LIF なしの 1 ステップ 122.9 ms のうち約 94 ms を占めていた。推論モードでは、tt-mlir が BN を conv に統合し、28.9 ms になる（S18）。
- 影響: S4〜S17 の p150a と A30 の比（例えば batch 8 で 25 倍）は、条件の揃っていない比較だった。S12 の `a30-hostcost.py`（8.27 ms）は学習モードのまま測定している（`model.eval()` を呼び出していない）。
- 扱い: `xla-sew18.py` に `BN=eval` を追加した（入力の batch の統計を running stats にコピーしてから推論モードにし、CPU の基準も推論モードにする）。今後の比較は推論モードで行う。A30 の基準も同じ条件で再測定する。

### D24. tt-xla 1.4.0 の level 1 は、小さいチャネル数の stem の conv を誤って計算する（実装）

- 期待: optimization_level 1 でも、conv は CPU とほぼ一致する（S10 で level 1 を使い始めたときは、SEW-ResNet18 の logits の相関だけを確認していた）。
- 実際: SEW-ResNet18 の stem の conv（3 → 64 チャネル、7×7、stride 2、padding 3、fp32、batch 8 と 32）は、level 1 では CPU との相関が 0.28 前後で、出力のばらつきは CPU の約半分になる。level 0 では 0.999998。入力のチャネルを 4、8、16 にゼロパディングしても誤り（0.21〜0.28）、32 なら正しい（1.000000）。ほかの conv（3×3 と 1×1、64〜512 チャネル）は level 1 でも 0.999997 以上（S18。`tools/xla-conv-check.py`、`tools/xla-stem-pad.py`）。
- 影響: S10〜S17 の level 1 の SEW-ResNet18 は、最初の層から誤った値で計算していた。logits の相関（0.85〜0.89）は、logits の大半が線形層のバイアスだったために高くなっており、誤りを示さなかった。時間は、計算量が同じなので大きくは変わらない見込み（stem の conv の時間が誤った設定のものだった可能性はある）。
- 扱い: `xla-sew18.py` に `STEM_PAD=32`（デバイスのモデルだけ、stem の conv の入力と重みを 32 チャネルにゼロパディングする）を追加した。1 ステップは約 4.6〜6.7 ms 増える。tt-xla の新しいバージョンで修正されているかは確認していない。GitHub への報告はユーザーが決める。
- 追記（S23）: stem を space-to-depth（stride folding）で 4×4、stride 1 の conv（入力 12 チャネル）に変換しても同じで、16 チャネルにゼロパディングすると CPU との相関 0.39、32 チャネルなら 0.999996。

### D25. tt-lang の演算の出力先を `torch.zeros_like` で生成すれば、呼び出しごとに別のバッファになると思った（実装）

- 期待: `lif_op(x, torch.zeros_like(x))` のように呼び出せば、呼び出しごとに新しい出力先が確保され、結果は呼び出しごとに別のテンソルになる（tt-xla の公式のテスト `tests/torch/ops/test_tt_lang_kernel_e2e.py` も出力先を `torch.zeros_like` で生成する。ただし 1 つのプログラムで呼び出すのは 1 回だけ）。
- 実際: tt-mlir は形状と値が同じ定数を 1 つに共通化する（SEW-ResNet18 の TTNN の IR で、20 個の LIF に対して `ttnn.full` は形状ごとの 5 個）。`ttnn.tt_lang_op` は DPS の演算で、TTNN では結果を返さない `ttnn.generic` になり、出力先の引数に直接書き込む。そのため同じ形状の LIF がすべて 1 つのバッファに書き込み、結果はそのバッファから読み出される。順に使い切る間は問題ないが、downsample のあるブロック（layer2.0、layer3.0、layer4.0）の `sn2 + downsample_sn` は `ttnn.add(%5, %5)` になり、後に書き込んだ `downsample_sn` の 2 倍になっていた。layer2 の出力で値 2 の割合が CPU の 0.44% に対して 2.74%、平均が 0.098 に対して 0.145 で、layer4 までで CPU との不一致は `SimpleLIFNode` の 14.4% に対して 28.4% だった（S19）。
- 影響: S14〜S18 の tt-lang の LIF を用いた SEW-ResNet18 の出力は、layer2 から誤っていた（LIF の入力のレイアウトによらない。`flat`、`nchw`、`nhwc4` のすべて）。logits の CPU との相関は 0.731 で、元の LIF の 0.908 より低かった。時間には影響しない（演算は同じ）。`xla-ttl-lif.py` のように 1 つの LIF の出力を次の LIF の入力にするだけの接続では、この誤りが表面化しない。
- 扱い: `xla-sew18.py` の `TtlLIF` は、LIF ごとに異なる値で埋めた出力先を渡す（`torch.full_like(x, -1.0 - k)`。カーネルは全要素を書き換えるので、値は結果に影響しない）。IR で `ttnn.full` が 20 個になり、各 `ttnn.generic` が別のバッファに書き込むことを確認した。logits の相関は 0.911（`SimpleLIFNode` は 0.908）、1 ステップは 43.5 ms で変わらない。`LIF_OUT=zeros` で元の生成方法に戻せる。tt-mlir の側で修正すべきもの（DPS の出力先にした定数を共通化しない、または `ttnn.generic` の前に出力先を新しく確保する）と見ている。新しいバージョンで修正されているかは確認していない。GitHub への報告はユーザーが決める。

### D26. A30 と p150a の SEW-ResNet18 を、同じ fp32 の精度で比較しているとした（判断）

- 期待: どちらも fp32 のモデルなので、A30 の基準（S4 の 14.8 ms）と p150a の時間は同じ精度の計算どうしの比較になる。
- 実際: SpikingJelly の `benchmark/benchmark_snn_single_gpu.py`（502〜503 行目）は `torch.backends.cuda.matmul.allow_tf32 = False` と `torch.backends.cudnn.allow_tf32 = False` を無条件に設定するので、A30 の conv は TF32 を用いない厳密な fp32 で計算している（PyTorch 2.11 の既定は `cudnn.allow_tf32 = True`）。一方 p150a の conv の出力は、仮数 10 ビットに丸められている（デバイスの stem の BN の出力の 99.84% が仮数 10 ビットに収まる。S19）。
- 影響: S18 の「p150a は A30 の約 2.9 倍」は、A30 だけが厳密な fp32 という、p150a に有利な条件での比較だった。TF32 を有効にすると、A30 は 14.75 ms から 7.5 ms になる（S20）。S12 の `a30-hostcost.py`（8.27 ms）が S4 より速かったのも、主に TF32 のため。
- 扱い: `tools/a30-sew18.py` で TF32 を有効にした時間を測定した（S20）。p150a（まとめて同期 31.5 ms）は、TF32 の A30 の 4.2 倍、厳密な fp32 の A30 の 2.1 倍。比較の基準は TF32 の A30 とする（ユーザーの決定、2026-10-28。ベンチマークのままの値は併記する）。

### D27. LIF のカーネルは読み書きが律速だとした（判断）

- 期待: LIF は要素ごとの単純な計算なので、メモリの読み書きが律速になる。layer1 の 150〜190 GB/s は、1 回に転送するブロックが 2 タイルと小さいため（S22 での推定。作業記録の「次にやること」の (b)）。
- 実際: layer1 の形状で行ごとのサイクル数を測定すると（`run-test.sh --auto-profile`）、読み出しのスレッドは 44.8% の時間、バッファの空きを待機しており、計算が律速だった。1 時刻 1 タイルに約 4,500 サイクル。生成された C++ では、tt-lang は `store` ごとに DEST の区間を分割し、区間ごとに `init_sfpu`、演算ごとに初期化を呼び出し、2 つの `store` が同じ部分式を用いても別々に計算する。`floor → relu → sign`（1 行あたり約 30 命令）を 1 時刻に 2 回計算していた。
- 影響: ブロックを拡大するだけでは不十分で、先に計算を削減する必要があった。計算を削減すると、今度は転送（1 コアにつき 1 件ずつ、完了を待ってから次の転送を発行する実装）が目立つようになった。
- 扱い: しきい値の判定を `lt` と `fill` で実装し、次の時刻へ h を引き継ぎ（`store` が 2 つでも再計算する量が小さい形）、ブロックを `(rb, bw)` に拡大した（S25）。tt-lang のカーネルは、生成された C++（`run-test.sh --emit-runner`）と `--auto-profile` で区間ごとのコストを確認してから設計する。計算のみ・転送のみのバージョンを作成して比較すると、どちらが律速かを切り分けられる。

### D28. `grid="auto"` は tt-xla の経路でも実機のコアの格子になると思った（実装）

- 期待: `@ttl.operation(grid="auto")` は、コンテナ（ttnn）と同じく、tt-xla の経路でも p150a の計算コアの格子 11×10（110 コア）に解決される。
- 実際: tt-xla 1.4.0 の `tt_torch/tt_lang.py` の `_make_deviceless_ttnn_args` は、仮の device（`_StubTtnnDevice(grid_cols=8, grid_rows=8, arch="wormhole_b0")`。注記に「この統合が対象とする 8×8 の Wormhole_b0」）を渡す。tt-lang の `_resolve_grid` は `"auto"` をその `compute_with_storage_grid_size()` で決定するので 8×8 になり、IR の `ttnn.generic` の `core_ranges` も (0,0)–(7,7) だった。`ttl.target_arch` も `wormhole_b0` になる。
- 影響: S24、S25 のモデルの LIF は 20 個とも 64 コアで実行されていた。コンテナのベンチマーク（110 コア）の時間は、モデルの中の LIF の時間を表していなかった。
- 扱い: `make_lif` と `make_lif4` に `grid` の引数を追加し、`xla-sew18.py` と `xla-ttl-lif.py` は `LIF_GRID=11,10` で渡す（S26）。IR の `core_ranges` が (0,0)–(10,9) になり、まとめて同期の 1 ステップが 10.54 → 10.18 ms。`arch` は修正していない（tt-lang 統合の中で決まる）。

### D29. コンテナで動作したカーネルが、tt-xla の経路の SEW-ResNet18 の中でデバイスをハングさせた（実装）

- 期待: コンテナ（tt-lang 1.1.10、ttnn）で正しく動作した tt-lang のカーネルは、tt-xla の経路（tt-lang 1.1.5、仮の Wormhole の device で構築、D28）でも動作する。
- 実際: `make_lif4(bf16_out=True)`（fp32 で計算し、スパイクを fp32 の CB から別の区間で bf16 に変換して書き込む。CB は fp32 が 3 つと bf16 が 1 つ）は、コンテナでは stem の形状でも CPU とビット単位で一致した（560 µs）。SEW-ResNet18（`STEM_SPIKE=bf16`、trace）では、グラフのコンパイルの後、最初の実行の途中でハングした（ログが 15:49:28 から進まず、1 本のスレッドが CPU を占有し続けた）。SIGINT では停止せず、SIGTERM で停止させた後、コンテナで小さい LIF を実行してもハングしたので、`tt-smi -r` で復旧した（2026-10-28 16:47）。
- 分かっていること: IR は意図どおり（LIF が bf16 の `%arg67` に書き込み、`max_pool2d` が直接読み出す。maxpool の前の `typecast` は除去された）。tt-xla が生成した計算カーネルは、ブロックのサイズを除けばコンテナのものと同じ構成で、区間ごとに `init_sfpu(入力, 出力)` でパッカーを再設定していた。LIF のカーネル、モデルの中での L1 の使用状況（CB は 1 コア約 850 KB）、maxpool のどれでハングしたかは切り分けていない。
- 扱い: `STEM_SPIKE=bf16` は使用しない。切り分けるには、bf16 の `make_lif4` を tt-xla の経路で単体で実行する試験と、`PREFIX=sn1` で maxpool の前まで実行する試験が必要（いずれもハングした場合はリセットが必要）。

### D30. tt-xla から conv の設定や L1 への配置を指定できない（実装）

- 期待: S28 で、tt-xla 1.4.0 のプラグイン（`pjrt_plugin_tt.so`）の文字列に、tt-mlir の `override-conv2d-config`、`override-output-layout`、`insert-memreconfig` が含まれていた。コンパイルのオプションから、これらか、L1 への配置に限った指定を渡せるかもしれない。
- 実際（S29。サブエージェントが手元の tt-xla の clone（`b360078`。1.4.0 は `941a6b8` のビルドで、clone の方が新しい）と 1.4.0 の `.so` を読んだ。実機では試していない）: コンパイルのオプションは `pjrt_implementation/src/api/compile_options.cc:16-142` の決まったキー（`optimization_level`、`enable_trace`、`math_fidelity`、`fp32_dest_acc_en`、`experimental_weight_dtype` など）のみで、未知のキーは警告なしに破棄する（`compile_options_parser.cc:349-351`）。tt-mlir に渡すのは `module_builder.cc:1016` の `optimizationLevel` などの決まった項目のみで、パイプラインのオプションをそのまま渡す経路はない。`.so` にある名前は、tt-mlir の定義がリンクされているだけ。level 0、1 は活性化をすべて DRAM に配置し、L1 に配置するのは level 2（tt-xla の `docs/source/performance.md:49-68`。conv を含むとコンパイルできない。D9、D15）。`experimental_weight_dtype` の対象は matmul と linear のみで、conv には適用されない（`performance.md:160`）。
- 分かっていること: コンテナの ttnn で conv の入出力を L1 に配置し、`enable_act_double_buffer` を用いると、stem を除く conv は 3.71 → 2.34〜2.74 ms、LIF（layer1〜4）は 1.27 → 0.92 ms になる（S29、単体の和）。tt-mlir が付与する設定（`act_block_h_override`、シャードの指定なし、double buffer なし、DRAM の interleaved）のままでは、この短縮分は得られない。
- 追記（S29、ネットワーク全体）: IR の演算の出力を連結して再実行する `ttnn-replay-chain.py` で、layer1 以降の活性化を L1 に配置し、conv に double buffer を用い、L1 に収まらないテンソルだけを DRAM に配置すると、1 ステップのデバイスの稼働時間は 9.84 → 7.97 ms になった（コンテナ。LIF のスパイクは全層で CPU と一致）。tt-xla の経路で得られていない短縮分は、少なくともこの 1.9 ms。
- 扱い: tt-xla の経路の中では、conv と活性化の配置先を調整できない。迂回策の候補は `backend=codegen_py`（TTNN の Python のコードを出力し、手作業で修正して `codegen_load_py` で読み込む。`compile_options.h:12-31`、`compile_options.cc:54-66`。p150a、trace、tt-lang の演算で動作するかは確認していない）か、モデルを ttnn で直接記述すること（公式の ResNet-50 と同じ方式）。

### D31. 1 ステップごとに同期するときの追加の時間を、tt-xla の実行時のコストだとした（判断）

- 前提: SEW-ResNet18 で、1 ステップごとに同期（S30 で 14.2 ms）がまとめて同期（9.59 ms）より約 5 ms 長いのは、trace の起動、出力のホストへの転送、ホストのディスパッチが、デバイスの計算と重ならないため（S24 の「次にやること」の (c)）。回ごとに 1〜2 ms ばらつく理由は分かっていなかった。
- 実際（S31）: 入力の大きさによらず、ごく小さいグラフでも、プロセスごとに 0.4 ms の回と 4 ms の回に分かれた。遅い回はスレッドの実際の周波数が 800 MHz（`acpi-cpufreq` の最低）で、Python の関数呼び出しまで遅い。SMT の兄弟の論理 CPU（156〜167）に SCHED_IDLE のビジーループを置き、計測を 60〜71 に固定すると、小さいグラフは 0.21 ms、SEW-ResNet18 は 10.99 ms（ばらつきもほぼ消えた）。待機するスレッドがスリープする（または `yield` する）ので、`schedutil` から見た使用率が低く、周波数が上がらず、アイドルからの復帰の遅延も毎回加わると見ている（C ステートは確認していない）。A30 の CUDA の同期はビジーウェイトで待機するので、ホストの CPU はアクティブなままになる（推測）。
- 教訓: ホストとデバイスの往復を含む時間がプロセスごとに二峰に分かれたら、デバイスより先に、ホストの CPU の周波数（`thread-sampler.py` の APERF/MPERF の値）を確認する。
- 扱い: 表の値は既定の省電力設定のまま測定している。`performance` の governor（sudo）か、計測時のビジーループを採用するかは、ユーザーが決める。採用するなら A30 も同じ条件で再測定する。

### D32. tt-xla に、入力をタイルレイアウトで渡せるかもしれないとした（実装）

- 前提: stem の入力は毎ステップ、デバイス上で行優先からタイルレイアウトに変換している（2 つで 0.86 ms。幅が 32 しかないので遅い）。ホストでタイルレイアウトに変換してから渡せばこの変換を排除できるが、比較条件の判断が必要（案の一覧の x3）。
- 実際（S33）: 別にコンパイルしたグラフの出力（デバイスの DRAM 上のタイルレイアウトのテンソル）を次のグラフに渡しても、次のグラフの trace の入力は `#system_memory` の行優先の型のままで、trace の中で `to_layout` を実行する。時間もホストの入力を直接渡す場合と同じだった。tt-xla 1.4.0 には trace の入力の型を変えるオプションもない（D30）。
- 扱い: tt-xla の経路では、入力のタイル化は排除できない。ttnn で直接記述すれば、入力を最初からタイルレイアウトで配置できる（案の一覧の t4）。
