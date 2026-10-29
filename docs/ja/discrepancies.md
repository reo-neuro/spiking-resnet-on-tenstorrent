# 前提と実際の食い違い

SNN の検証で、前提にしていたことと実際が違っていたものを記録する。同じ思い込みを繰り返さないためと、TT や SpikingJelly の制約を一か所で引けるようにするためである。

種類は 2 つある。

- 判断: エージェント（またはユーザー）が確かめずに置いた前提が、実装や実測と違っていたもの
- 実装: 公式の文書や一般的な期待と、TT や SpikingJelly の実際の動作が違っていたもの

新しい項目は末尾に D 番号で足し、番号は振り直さない。解決したら「扱い」を書き換え、消さない。確定した制約は `setup.md` にも書く。経緯の詳細は作業記録の S 番号で引く（作業記録は公開していない）。

## 判断の教訓

- 回避策を勧める前に、TT 側の実装（tt-xla のテスト、公式のベンチマークでの使い方）を読む（D3）
- 「動かない」と結論する前に、失敗したテストが何を流していたか（forward だけか、backward もか）を確かめる（D4）
- 公式が推奨する実行経路（PyTorch なら `torch.compile(backend="tt")`）を先に試す。別の経路で見つけた問題は、公式の経路で再現するかを見てから一般化する（D5）
- ライブラリの使い方は手元にある版の文書とコードで確かめる。記憶にある古い版の API で説明しない（D1、D2）
- 「固定費」は入力の大きさにも作り方にもよらず、本質的に削れないものにだけ使う。batch によらないだけのもの（演算の数に比例する送り出しの費用など）を固定費と呼ばない（D16）
- 移植の作業ではライブラリの中心の部品（SNN なら LIF）を先にターゲットのカーネルにする計画を立てる。周辺の設定（最適化レベル、trace）の調査から入らない（D16）
- ローカルの clone にある skill や文書を「公式の今のもの」として示す前に、upstream に同じ場所で残っているかを確かめる（D17）
- ベンチマークの条件（学習モードか推論モードか、精度、batch）は、ベンチマークのコードの該当行で確かめる。比べる相手と同じ条件かを、数値を比べる前に確かめる（D23）
- 正しさを網の最後の出力の相関だけで判断しない。出力がバイアスに引きずられていると、途中が壊れていても相関は高く出る。層ごとに CPU と比べる（D24）
- 正しさの指標が下がったとき、精度（丸め）のせいと決める前に、層ごとに出力の分布（スパイクの率、値の種類の割合）を CPU と比べる。丸めはしきい値の近くの反転を散らすだけで、分布はほとんど変えない。分布が変わるなら別の誤りがある（D25）
- SEW-ResNet18 の logits の相関は、どこかに 1e-5 以上の乱れが入ると 0.92 前後まで下がって止まる（推論モードの BN、stem を埋めた版。S19）。小さな精度の差の大小は測れず、それより大きく下がったときだけ系統的な誤りを示す
- 時間を測るときは、デバイスの完了まで待っているかを、出力をホストに読み戻す版と比べて確かめる。tt-xla の `torch_xla.sync(wait=True)` は待たない。カーネルの帯域はデバイスのプロファイラのカーネルの区間から出し、呼び出し 1 回の壁時計から出さない（D21）
- ホストとデバイスの往復を含む時間が、同じ条件でプロセスごとに二峰に分かれたら、デバイスを疑う前にホストの CPU の実際の周波数を見る（D31）

## 一覧

| D | 種類 | 内容 | 扱い | 経緯 |
| --- | --- | --- | --- | --- |
| D1 | 判断 | SpikingJelly の GPU の比較相手を cupy カーネルとした | 2.0 では Triton かネイティブ CUDA。訂正済み | Web 検索 |
| D2 | 判断 | multi-step に `backend='torch'` を付ければ純 PyTorch で動くとした | 2.0 に `backend` 引数はなく、xla では `SimpleLIFNode` を使う | S2 |
| D3 | 判断 | 推論の BN を手で Conv に畳み込むことを勧めた | TT のコンパイラ自身が畳み込む。BN は eval のまま渡す | S3 |
| D4 | 判断 | 「eval モードの BN はコンパイルできない」とした | backward も流すテストだったため。forward は動く | S3 |
| D5 | 判断 | lazy tensor の経路だけで検証を進めた | 公式は `torch.compile(backend="tt")`。学習時の誤りは公式の経路で未確認 | S3 |
| D6 | 実装 | `LIFNode` などの登録済みのニューロンが xla で動かない | `SimpleLIFNode` を使う | S2 |
| D7 | 実装 | Conv の直後の `nn.BatchNorm2d`（学習モード）の backward が壊れる | `ManualBN2d` で回避。原因は未特定 | S2 |
| D8 | 実装 | Conv の後の `nn.GroupNorm` の入力勾配もずれる | 使っていない。原因は未特定 | S2 |
| D9 | 実装 | optimization_level 1・2 で、Conv を含むグラフがコンパイルできない | 1.5.0.dev20260831 の退行。1.4.0 なら level 1 は通る | S2、S3、S8 |
| D10 | 実装 | A30 で CLIF の Triton カーネルの境界値のテストが 1 件落ちる | LIF には関係しないので保留 | S1 |
| D11 | 判断 | `SimpleLIFNode` は single-step の作りなので、T のループを外で回す必要があるとした | `multi_step_forward` があり、multi-step のまま使える | S4 |
| D12 | 実装 | SEW-ResNet18 が batch 16 以上で実行時に止まる | batch 8 までで測る。1.4.0 でも起き、batch 12 でも止まる。tt-mlir #8559（未解決）と同じ形と見ている | S4、S9、S10 |
| D13 | 判断 | TT の index に、今の版より新しい pjrt-plugin-tt があるとした | 公開の最新は今の版（08-31）。新しい tt-mlir を使うにはソースからのビルドが要る | S5 |
| D14 | 判断 | SEW-ResNet18 の固定のコスト（約 315 ms）は、演算を 1 つずつホストから dispatch する時間だとした | デバイス上の時間が batch 8 で 243 ms あり、batch に比例する | S6 |
| D15 | 実装 | trace が optimization_level 0 では Conv を含むとコンパイルできない | 小さいモデル（Conv＋BN＋ReLU）では 1.4.0 の level 1・2 でも abort する。ただし SEW-ResNet18（level 1、推論モードの BN）は trace で動く（S23） | S6、S8、S23 |
| D16 | 判断 | 演算の送り出しの費用を「固定費」と呼び、LIF のカーネル化を後回しにしてプラグインの設定の調査を続けた | 送り出しの費用は演算の数に比例し、LIF の融合で減らすのが本筋。LIF のカーネル化を中心の作業にする | S11、S12 |
| D17 | 判断 | 手元の `~/tt-metal` の `.agents/skills` を、今の公式の skill として示した | upstream では 09-01 に削除され、`tenstorrent/skills` に移った。取り込み直し済み | S13 |
| D18 | 実装 | 同じ tt-lang のカーネルを 1 つのグラフで 2 回以上呼ぶと、2 回目の resolve が失敗する | tt-lang のカーネルごとのキャッシュで、tt-xla の横取りが効かない。呼ぶたびにカーネルを作り直して回避 | S14 |
| D19 | 実装 | tt-lang の仕様（手元の clone、09-30）の書き方が、tt-xla 1.4.0 の wheel の tt-lang 1.1.5 では通らない | `>=`、`where`、`mask`、`clamp` がなく、fp32 の CB の制約もある。`sign(relu(floor(h)))` などで組んだ | S14 |
| D20 | 実装 | 同じ条件の 1 ステップの時間は一定だとした | デバイスに 0.3〜0.4 秒続けて送ると最大 2 倍遅くなり、3 秒休むと戻る。出力を読み戻して待つと消えるので、ホストがデバイスの先を走る分と見ている（S16、D21） | S15、S16 |
| D21 | 判断 | `torch_xla.sync(wait=True)` までの時間を、デバイスの処理の時間として扱った | デバイスの完了を待たない。LIF の 1 回は 30.5 µs と出たが、読み戻して待つと 1,315 µs。S4〜S15 の 1 ステップの時間は、ホストの送り出しの時間か、デバイスに追いつかれた後の時間 | S16 |
| D22 | 実装 | tt-lang のカーネルの fp32 の計算は fp32 の精度で行われる | 2 つの CB を足すと FPU で仮数約 10 ビットに丸める。SFPU の経路なら正確だが、それには CB ごとの `UnpackToDestFp32` が要り、tt-xla 1.4.0 の橋渡しはそれを運ばない。tt-xla の経路では CPU とビット単位で一致させられない。読み込みは切り捨てで、SEW-ResNet18 への影響は conv の丸めより 2 桁小さい | S16、S17、S19 |
| D23 | 判断 | SpikingJelly のベンチマークは推論でも `model.eval()` を呼ばないので、BN は学習モードだとした（S4） | 推論の段は `model.train(args.phase == "training")` で推論モード。A30 は推論モード、p150a は学習モードの BN で比べていた。学習モードの BN と前後の並べ替えが、LIF なしの 123 ms のうち約 94 ms を占めていた | S4、S18 |
| D24 | 実装 | tt-xla 1.4.0 の optimization_level 1 は、conv を正しく計算する | 入力が 3、4、8、16 チャネルの stem の conv（7×7、stride 2）だけ、CPU との相関が 0.21〜0.29 になる。32 チャネルにゼロで埋めると正しい。level 1 の SEW-ResNet18 の出力は S10〜S17 のあいだ誤っていた | S18 |
| D25 | 実装 | tt-lang の演算の出力先を `torch.zeros_like` で作れば、呼び出しごとに別のバッファになる | tt-mlir は同じ形の 0 の定数を 1 つにまとめ、カーネルはそこに直接書くので、同じ形の LIF がすべて 1 つのバッファを共有する。2 つの LIF の出力を足すブロックが壊れ、logits の相関が 0.908 ではなく 0.731 になった。LIF ごとに別の値で埋めた出力先を渡して回避 | S19 |
| D26 | 判断 | A30 と p150a の SEW-ResNet18 を、同じ fp32 の精度で比べているとした | A30 の基準（SpikingJelly のベンチマーク）は TF32 を切った厳密な fp32 の conv、p150a の conv は仮数 10 ビット相当。p150a に有利な条件の比較になっている。TF32 を許すと A30 は 14.75 ms から 7.5 ms | S19、S20 |
| D27 | 判断 | LIF のカーネルは、1 回に運ぶブロックが小さいので読み書きが律速だとした（S22 の見立て） | SFPU の計算が約 7 割だった。tt-lang は `store` ごとに区間を分けて初期化を繰り返し、2 つの `store` に共通の部分式も別々に計算する。閾値を `lt` と `fill` にして計算を減らし、ブロックを広げて、20 層 2.81 → 1.92 ms | S25 |
| D28 | 実装 | tt-lang のカーネルの `grid="auto"` は、tt-xla の経路でも実機の計算コアの格子（p150a は 11×10）になる | tt-xla 1.4.0 の橋渡しは、仮の 8×8 の Wormhole の device でカーネルを組み立てるので、64 コアになる（`arch` も `wormhole_b0`）。S24、S25 のモデルの LIF は 64 コアで動いていた。grid を明示すれば 110 コアで動く | S26 |
| D29 | 実装 | コンテナで正しく動いた tt-lang のカーネルは、tt-xla の経路でも動く | stem の LIF に bf16 のスパイクを書かせる版（`make_lif4(bf16_out=True)`）は、コンテナでは正しく動いたが、SEW-ResNet18 の中ではデバイスが止まり、`tt-smi -r` が要った。原因は分かっていない | S26 |
| D30 | 実装 | プラグインに tt-mlir の `override-conv2d-config` があるので、tt-xla から conv の設定や L1 への置き方を渡せるかもしれない | 1.4.0 が受けるのは決まったキーだけで、知らないキーは黙って捨てる。level 0、1 は活性化を全部 DRAM に置く。L1 に置き、conv の double buffer を使えば、stem を除く conv と LIF で約 1.3 ms 縮む（コンテナの単体）が、tt-xla の経路では使えない | S29 |
| D31 | 判断 | 1 ステップごとに同期するときの上乗せ（約 5 ms）は、tt-xla の実行時の費用（trace の起動、読み戻し、送り出し）だとした | 大半はホストの CPU の省電力。待つたびにスレッドが眠り、CPU は 800 MHz のまま動く。SMT の兄弟を空回りさせると 14.2 → 11.0 ms。直すかはユーザーが決める | S24、S31 |
| D32 | 実装 | tt-xla で、入力をタイルの形で渡せるかもしれない（ホストでタイルにするかは比べる条件の判断） | tt-xla 1.4.0 の trace の入力は常にホストの行優先として扱われ、デバイスのタイルを渡しても trace の中でタイルにする。方法がない | S24、S33 |

## 各項目

### D1. GPU の比較相手は cupy カーネルとした（判断）

- 前提: SpikingJelly の CUDA の高速化は cupy のバックエンドで行う（エージェントの記憶。0.0.0.0.14 までの構成）
- 実際: 2.0.0（2026-10-17）のバックエンドは torch、Triton、ネイティブ CUDA。README に「no CuPy dependency required」とある（Web 検索で確かめた）
- 影響: 比較の基準を Triton とネイティブ CUDA に改めた
- 扱い: 訂正済み

### D2. `backend='torch'` で純 PyTorch の経路になるとした（判断）

- 前提: multi-step のニューロンに `backend='torch'` を付ければ、カスタムカーネルを通らない（エージェントの記憶。旧版の API）
- 実際: 2.0.0 の `LIFNode` に `backend` 引数はなく、バックエンドはテンソルのデバイスから自動で選ばれる。CPU でも xla でも、`torch.library` のカスタム演算子を通る（D6）
- 影響: 最初の実行で `LIFNode` が止まった。純 PyTorch の `SimpleLIFNode` に切り替えた
- 扱い: 訂正済み

### D3. 推論の BN を手で Conv に畳み込むことを勧めた（判断）

- 前提: eval モードの BN はコンパイルできないので（D4）、推論では BN を Conv に畳み込むのがよい（一般的な推論の最適化からの推測。TT 側は見ていなかった）
- 実際: tt-xla の `tests/torch/graphs/fusion_patterns/test_batch_norm_decomposition.py` によると、Conv の後の eval の BN はコンパイラが `alpha*x + beta` に分解し、`Conv2dWithMultiply` の融合で Conv の重みに畳み込む（optimization_level 1 以上）。公式の ResNet-50 のベンチマークも BN を残したまま動かしている
- 影響: ユーザーに指摘されて確かめた。手で畳み込むとコンパイラの仕事と重複する
- 扱い: BN は eval のまま渡す。ただし融合が効く 1・2 は今は使えない（D9）

### D4. eval モードの BN はコンパイルできないとした（判断）

- 前提: `tools/xla-op-backward.py` の `bn_eval` が `The size of mean should be compatible with feature count` で失敗したので、eval の BN は TT で使えない
- 実際: そのテストは eval の BN にも backward を流していた。forward だけの `tools/xla-bn-eval.py` では、lazy でも `torch.compile` でも動く（optimization_level 0、相関 0.99991）
- 影響: D3 の誤った推奨の根拠になった
- 扱い: 訂正済み。eval の BN の backward は今も失敗するが、使う場面はない

### D5. lazy tensor の経路だけで検証を進めた（判断）

- 前提: テンソルを xla デバイスに載せて eager に書けば、torch_xla の標準的な使い方になる（blacksmith の MNIST がこの形だった）
- 実際: tt-xla の文書と公式のベンチマークは `torch.compile(..., backend="tt")` を使う
- 影響: D7、D8、D9 は lazy の経路で見つけたもの。D9 は `torch.compile` の経路でも再現した。D7 と D8 は `torch.compile` の経路ではまだ確かめていない
- 扱い: 以降は `torch.compile(backend="tt")` を主にする

### D6. 登録済みのニューロンが xla で動かない（実装）

- 期待: SpikingJelly 2.0 の README では、torch の参照実装が CPU と、拡張がないときの経路になっている。純 PyTorch なら xla でも動くと考えた
- 実際: 2.0.0 のニューロンは `torch.library` のカスタム演算子として、CPU と CUDA のディスパッチキーにだけ実装が登録されている（`ops/dispatch.py`）。xla では `RuntimeError: SpikingJelly neurons support CPU and NVIDIA CUDA, not xla:0`。`functional.neuron_implementation` も `ValueError: Unsupported neuron device xla:0`
- 影響: SEW-ResNet18 などのモデルはニューロンを `SimpleLIFNode` に差し替えないと動かない
- 扱い: `SimpleLIFNode` を使う。参照実装を XLA のキーにも登録するパッチを当てる案もある（未着手）

### D7. Conv の直後の学習モードの BN の backward が壊れる（実装）

- 期待: 単独で正しい演算子は、組み合わせても正しい
- 実際: Conv と BN はそれぞれ単独なら CPU と合う（相関 0.999 以上）。Conv → BN にすると、Conv の重みの勾配の最大値が CPU の 535 に対して 51,400 になる（相関 0.11）。精度のオプションを変えても結果は同じ
- 影響: そのままでは学習の loss が CPU から離れていく
- 扱い: `tools/sj_xla.py` の `ManualBN2d` で回避。原因は未特定（Conv の出力の NHWC と BN の backward の軸の取り違えを疑っている）。`torch.compile` の経路では未確認

### D8. Conv の後の GroupNorm の入力勾配がずれる（実装）

- 期待: D7 の回避策として GroupNorm が使える
- 実際: Conv → `nn.GroupNorm(4, 16)` で、入力の勾配の相関が 0.90、Conv の重みの勾配の相関が 0.58
- 影響: 正規化の層を変換するところに共通の問題がある可能性がある
- 扱い: 使っていない。原因は未特定

### D9. optimization_level 1・2 で Conv を含むグラフがコンパイルできない（実装）

- 期待: 公式の ResNet-50 のベンチマークは optimization_level 2 で動いている
- 実際: pjrt-plugin-tt `1.5.0.dev20260831000501` で、1 は `ttnn.permute failed validation`（`vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)`）、2 はコンパイラ（`libTTMLIRCompiler.so`）の中で segfault。lazy でも `torch.compile` でも同じ
- 影響: BN の融合などの最適化が使えず、p150a の速さは公式の条件より不利になる
- 扱い: 0 で測り、比較には但し書きを付ける。新しいプラグインで直っているかを見る（作業記録の「次にやること」）
- 追記（S6）: Conv 1 個（16→32、3×3）でも、bf16 でも、公式の例 `examples/pytorch/mnist_performant.py`（level 2、trace）でも起きる。モデルの組み方の問題ではない。segfault のスタックは `OpModel<PermuteOp>::getOpConstraints` → `ttnn::permute` → `ProgramImpl::update_kernel_groups` で、1 の `_M_range_check` と同じ、コンパイラが制約を問い合わせる経路の中。このプラグインは sfpi 7.69.0 を想定しているが、ホストの `/opt/tenstorrent/sfpi` は 7.83.0（2026-10-22 に 7.72.0 から上がった）
- 追記（S7）: 7.69.0 を venv の中（`pjrt_plugin_tt/tt-metal/runtime/sfpi`）に置いても、level 1 の Conv 1 個は同じ `_M_range_check` で、`mnist_performant.py` は同じ `update_kernel_groups` で落ちた。level 1 のエラーはカーネルのビルドより前に出ている。sfpi の版は原因ではない
- 追記（S8）: 同じマシンで、プラグイン 1.4.0（`~/tt-xla-venv`、07-23 ビルド）なら level 1 の Conv（単独でも、eval の BN 付きでも）が通る。blacksmith の venv（1.5.0.dev20260831000501、`.so` は `venv-xla` と同一）では同じく落ちる。**D9 はこのマシンの問題ではなく 1.5.0.dev20260831000501 での退行**である。1.4.0 でも level 2 は、Conv＋BN 1 組のコンパイルが 28 分たっても終わらなかった（CPU 100%、mock device で探索中。SIGINT が効かず SIGTERM で止めた。デバイスは正常のまま）

### D10. A30 で CLIF の境界値のテストが落ちる（実装）

- 期待: リリースの検証を通った 2.0.0 なので、テストは全部通る
- 実際: `test_clif_scalar_division_matches_torch_at_threshold[1.03]` が落ちる。しきい値ちょうどの入力で、Triton 版と torch 版のスパイクが 3 要素中 2 つずれる。上流のリリースの検証（PR #778）で GPU として挙がっているのは RTX 4090 で、A30（sm_80）で検証されているかは確かめていない
- 影響: LIF には関係しない
- 扱い: 保留。CLIF を使うときに調べる

### D11. `SimpleLIFNode` は single-step の作りとした（判断）

- 前提: S2 の手順2のスクリプトで 1 ステップずつ呼んでいたので、`SimpleLIFNode` は single-step 専用だと思い込んだ（コードを読んでいなかった）
- 実際: `SimpleBaseNode.multi_step_forward` が `torch.stack([self.single_step_forward(x) for x in x_seq])` で T のループを回す。`step_mode="m"` で `[T, N, ...]` を受け取れる
- 影響: S3 で「T のループを Python で回してグラフを展開させる」という余計な案を出していた
- 扱い: 訂正済み。SEW-ResNet18 は multi-step のまま組んだ。Conv は T×N をまとめたバッチで計算され、A30 と同じ形になる

### D12. SEW-ResNet18 が batch 16 以上で止まる（実装）

- 期待: batch 8 で動くなら、メモリが足りる範囲で batch を増やせる
- 実際: batch 16（T×N = 64）と 32 で、実行時に `TT_THROW: Invalid subtile broadcast type`（`binary_ng_device_operation.cpp:224`、`ttnn::add`）。コンパイルは通る。デバイスは正常のまま
- 影響: A30 の batch 32（51.1 ms）と同じ条件では比べられない。batch 8 で比べた
- 扱い: どの加算か（SEW の ADD、BN、LIF の更新）は特定していない。新しいプラグインで直っているかを見る

### D13. TT の index に新しい pjrt-plugin-tt があるとした（判断）

- 前提: nightly の wheel は毎日出ているので、2026-08-30 ビルドの今の版より新しいものが TT の index にある（S4 の「次にやること」。index を引いていなかった）
- 実際: 2026-10-27 に `pip index versions pjrt-plugin-tt --pre --index-url https://pypi.eng.aws.tenstorrent.com/` を引くと、最新は `1.5.0.dev20260831000501`（今の版）。GitHub の tt-xla のリリースも 08-31 が最後。pypi.org には無い。tt-xla の main は 10-14 まで更新が続いていて、`third_party/CMakeLists.txt` の `TT_MLIR_VERSION` は `c282803f`（tt-mlir の PR #9279。while 対応の #9187 より後）
- 影響: 「新しい版を別の venv に入れる」手順はそのままでは進められない
- 扱い: 新しい版を試すなら、tt-xla をソースからビルドする（`docs/source/getting_started.md` の手順。tt-mlir のツールチェーンも作る）。進めるかはユーザーが決める

### D14. 固定のコストはホストの dispatch だとした（判断）

- 前提: batch 2 と 8 の壁時計（329 ms と 370 ms）の差が小さいので、時間の大半はバッチによらない固定のコストで、その中身は演算を 1 つずつホストから dispatch する時間である（S4。壁時計の 2 点だけからの推測）
- 実際: デバイスのプロファイラで見ると、1 回の実行は 1,708 個のプログラムで、デバイス上のカーネル時間の合計は batch 2 で 70 ms、batch 8 で 243 ms（batch にほぼ比例し、1 サンプルあたり約 29 ms）。batch 8 の壁時計の 3 分の 2 はデバイス上の計算。batch 2 ではデバイスの時間は壁時計の 2 割ほどで、残りはホスト側（dispatch など）と見られる
- 影響: 「trace で dispatch を消せば速くなる」の効き目は小さい。batch 8 以上ではデバイス上の計算が律速で、1 サンプルあたり約 29 ms は A30（batch 32 で 1.6 ms）の約 18 倍
- 扱い: 訂正済み。2 点の壁時計から内訳を推測しない。デバイスの時間を測ってから言う

### D15. trace が optimization_level 0 で Conv を含むとコンパイルできない（実装）

- 期待: trace（`enable_trace`、`TT_RUNTIME_TRACE_REGION_SIZE`）は optimization_level によらず使える（`docs/source/performance.md` に level の条件は書かれていない）
- 実際: level 0 で、Linear→ReLU→Linear は trace で動く。Conv→ReLU（fp32 でも bf16 でも）はコンパイルで `error: Memory layout is required for non-SystemMemory buffer type.` を出して abort する。SEW-ResNet18 は `Device-resident input must be on device, but found on system memory` で止まる。公式のテストで trace を使うものは、どれも level 1 以上だった。level 1 は D9 で Conv が通らない
- 影響: Conv を含むモデルで trace を使う手段がない
- 扱い: 新しいプラグインで見る。デバイスは毎回正常（`fw_id=0`）
- 追記（S8）: 1.4.0 でも、level 1 の trace と level 2 の trace（bf16）で、Conv＋BN は同じエラーで abort する。版によらない
- 追記（S23）: 1.4.0 の level 1 で、SEW-ResNet18（推論モードの BN、stem を 32 チャネルに埋めた版、tt-lang の LIF）は trace で動いた（1 ステップごとに同期 22.7 ms、まとめて同期 17.1 ms）。公式の push テストにも ResNet-50 を level 1 の trace で流すもの（`tests/torch/models/resnet/test_resnet.py` の `test_torch_resnet_inference_trace`）がある。小さいモデルとの違い（Conv の bias、入力が直接 Conv に入ることなど）は切り分けていない。「Conv を含むと trace は使えない」は誤りだった

### D16. 送り出しの費用を「固定費」と呼び、LIF のカーネル化を後回しにした（判断）

- 前提: batch を変えても時間が変わらない部分を「固定費」と呼んだ。また、LIF を ttnn か tt-metal で書き直すのは、torch_xla の参照実装の結果を見てから決めればよいとした（`setup.md` の冒頭の方針）
- 実際: その部分は演算をデバイスに 1 つずつ送り出す費用で、演算の数に比例する（1 個あたり約 150〜170 µs、S11）。最大の出どころは LIF を小さな要素ごとの演算に分けて流していること。A30 でも、LIF を Triton の 1 カーネルにした `LIFNode` は `SimpleLIFNode` より約 6 割速い（S12）。SNN のライブラリを移植するなら、LIF をカーネルにするのが中心の作業になるのは、始める前から分かることだった（ユーザーの指摘）
- 影響: S5〜S12 の多くを、プラグインの版、最適化レベル、trace の調査に使った。そこで分かったこと（D9 の退行、1.4.0 の level 1、送り出しの費用の大きさ）は、LIF のカーネル化の前提として使える
- 扱い: 用語を直した。次にやることの 1 番を LIF の融合カーネルの実装にした

### D17. upstream から消えた skill を、今の公式の skill として示した（判断）

- 前提: 手元の `~/tt-metal` の `.agents/skills`（graph-fusing、optimize、tt-lang、tt-device-usage など）を、Tenstorrent の今の公式の skill として示した（S13 の前半）
- 実際: upstream の tt-metal からは 2026-09-01 に削除されていた（PR #55086。専用のリポジトリへ移すため）。今は `tenstorrent/skills` にプラグインとしてまとめられている。tt-lang の skill は移っておらず、後継は tt-lang のリポジトリの `claude-slash-commands/` だった。ユーザーが upstream から取り込めないかと尋ねて分かった
- 影響: 古い版を読んで手段を比べた。graph-fusing の本体は upstream と同じで、比べた結論は変わらない
- 扱い: upstream から取り込み直した（`setup.md` の「公式の skill と slash command」）。教訓: ローカルの clone にある skill や文書を「公式の今のもの」として示す前に、upstream に同じ場所で残っているかを確かめる

### D18. 同じ tt-lang のカーネルを 2 回以上呼ぶと resolve が失敗する（実装）

- 期待: `@tt_torch.tt_lang_operation` で包んだカーネルは、グラフの中で何回呼んでもよい
- 実際: 2 回目の resolve で `TTLangError: tt-lang compile did not produce a CompiledTTNNKernel`。tt-lang（`ttl_api.py`）がカーネルのオブジェクトごとに、形をキーにしてコンパイル結果をキャッシュしている。キャッシュに当たると `_compile_kernel` が呼ばれず、tt-xla（`tt_torch/tt_lang.py` の `_drive_ttl_compile`）が横取りする関数が何も捕まえない。tt-xla の文書は「tt-lang は自分でキャッシュすべき」とだけ書いていて、この組み合わせは想定していないように見える
- 影響: SEW-ResNet18 のように、同じ形の LIF が何層もあるモデルで止まる
- 扱い: `@tt_lang_operation` の中で、呼ぶたびに `make_lif(T)` を作り直して回避（中身はグラフのコンパイル時にしか呼ばれない）。tt-xla の新しい版で直っているかは見ていない

### D19. tt-lang の仕様の書き方が、wheel の tt-lang 1.1.5 では通らない（実装）

- 期待: 手元の tt-lang の clone（09-30）の仕様と `/ttl-import` の手順どおりに書けば、tt-xla 1.4.0 の wheel の tt-lang（1.1.5.dev20260704）でも通る
- 実際: 次の点が違った。`>=` がない（`<`、`>`、`==`、`!=` はある）。`ttl.block.where`、`mask`、`clamp` がない（コンテナの 1.1.10 でもシミュレータにしかない）。fill は `ttl.math.fill` ではなく `ttl.block.fill`。カーネルの中で条件式（`a if c else b`）を使えない。キャプチャした `torch.dtype` を符号化できない（1.1.10）。fp32 では 1 つの CB を FPU の演算（add、sub）と SFPU の演算（mul、floor、スカラーとの乗算など）の両方で読めない
- 影響: LIF の式を組み直す必要があった（S14）
- 扱い: スパイクを `sign(relu(floor(h)))`、リセットを `v = hb · (1 − s)` にし、h を SFPU 用と FPU 用の 2 つの CB に書いた。tt-lang で書くときは、仕様ではなく使う版の `ttl/_generated_elementwise.py` と `ttl/operators.py` で使える演算を確かめる
- 追記（S25）: fill は、モジュールの頭で `fill = ttl.block.fill if hasattr(ttl, "block") else ttl.math.fill` と別名にすれば、1.1.5 と 1.1.10 の両方で使える（1.1.5 には `ttl.block` がなく、1.1.10 は `ttl.math.fill` を拒む）。型は、カーネルの中に `dtype=torch.float32` と直接書けば通る（キャプチャした変数の `torch.dtype` は符号化できないが、モジュールの属性として書いたものは通る）。1.1.5 でも、スカラーとの演算は掛け算だけ（足し算と引き算は 2 つのブロックの間だけ）
- 追記（S26）: 1 つの計算の区間に fp32 と bf16 のタイルが混ざると、どちらの版もエラーにする（`mixed f32 and non-f32 tile arguments`）。例外は fp32 の入力がどれも直接 `typecast` に渡るとき。1.1.10 では fp32 → fp32 の `typecast` を挟めば通ったが、1.1.5 ではそれが先に畳まれて通らない。fp32 の CB に書いてから、別の区間で fp32 → bf16 の `typecast` をかければ両方で通る（`make_lif4(bf16_out=True)`）

### D20. 同じ条件でも 1 ステップの時間が一定にならない（実装）

- 期待: 同じモデル、同じ入力を続けて流せば、2 ステップ目以降の時間はほぼ一定になる（S4〜S14 は中央値をそのまま使った）
- 実際: SEW-ResNet18（batch 8、level 1）は、休んだ直後の 2〜3 ステップが 52〜88 ms で、続けて回すと 110 ms（LIF なし）または 166 ms（tt-lang の LIF）になる。デバイスが 3 秒休むと戻る。batch 2 では 76 ms のまま 87 ステップ続き、その後 43.5 ms に速くなった。AICLK、電力、温度、Doppler の NOP、ホストの CPU の周波数、NUMA、ほかのプロセスでは説明できない（S15）
- 影響: 速さの比較が、休んだ直後の値か続けて回したときの値かで最大 2 倍変わる
- 扱い: 原因は分かっていない（調べるには perf か turbostat が要る）。追っても得るものが少ないとユーザーが判断し、調べるのを止めた（2026-10-27）。休んだ直後の値と続けて回したときの値を分けて報告する。計測は `taskset -c 60-71,156-167` で NUMA 5 に固定する（README の決まり。この現象には効かなかった）
- 追記（S16）: 毎ステップ出力をホストに読み戻して待つと（`WAIT=cpu`）、SEW-ResNet18 の 2 つの水準は消え、2 ステップ目から一定になった（tt-lang の LIF で 180〜186 ms、LIF なしで 122〜128 ms）。`sync(wait=True)` はデバイスの完了を待たないので（D21）、休んだ直後の水準はホストが送り出す速さ、続けて回したときの水準はホストがデバイスに追いつかれた後の速さだと見ている（推測。ホストが先を走れる量に上限があり、デバイスが 3 秒休むとその分が空く）。batch 2 で 87 ステップ目から速くなったことと、コンテナの ttnn の試験（1 回ごとに同期）の 2 つの水準は、これでは説明できない。今後の SEW-ResNet18 の計測は `WAIT=cpu` で行う

### D21. `torch_xla.sync(wait=True)` までの時間をデバイスの時間として扱った（判断）

- 期待: `torch_xla.sync(wait=True)` は、送り出したグラフがデバイスで終わるまで待つ。S4〜S15 は、`m(x)` から `sync(wait=True)` までを 1 ステップの時間とし、S14 は K=1 の同じ時間（2.05 ms）からカーネルの帯域（約 100 GB/s）を出した
- 実際: tt-xla 1.4.0 で `[4·6272, 1024]` の tt-lang の LIF を K 回つなぐと、`sync(wait=True)` だけでは 1 回 30.5 µs で、デバイスのカーネル時間（コンテナのプロファイラで 737 µs）より短い。最後の出力を `.cpu()` で読み戻すと 1 回 1,315 µs（S16）
- 影響: S4〜S15 の 1 ステップの時間は、デバイスが先を走るホストに追いつくまではホストの送り出しの時間で、追いついた後はデバイスの時間に近い（D20）。S14 のカーネルの帯域（約 100 GB/s）は誤りで、コンテナのプロファイラでは 279 GB/s。S6 のデバイスのプロファイラの値は、この影響を受けない
- 扱い: `xla-ttl-lif.py` と `xla-sew18.py` に `WAIT=cpu`（毎回出力を読み戻す）を足し、以後の計測はこれで行う。カーネルの時間は `tools/ttl-lif-bench.py`（デバイスのプロファイラ）で測る

### D22. tt-lang の fp32 の計算は fp32 の精度で行われると思った（実装）

- 期待: fp32 のテンソルを受ける tt-lang のカーネルは、加減乗除を fp32 で計算する
- 実際: 2 つの CB を直接足す（`(v + x) * 0.5` など）と FPU で計算され、fp32 の入力が仮数約 10 ビットに丸められる（`x + x` で正確に一致した要素が 0、相対誤差は最大 9.4e-4。S16）。SFPU を通る書き方（`v + (x * 0.5 - v * 0.5)` など）なら、ttnn で直接動かすと CPU とビット単位で一致する（コンテナの 1.1.10。SEW-ResNet18 の LIF の 5 つの形で不一致 0。S17）。これは SFPU が直接読む CB に tt-lang が `UnpackToDestFp32` を設定するため（wheel の 1.1.5 にもある。`ttl/ttl_api.py` の `_set_unpack_to_dest_fp32`）。しかし tt-xla 1.4.0 の橋渡し（`tt_torch/tt_lang.py` の `_serialize_kernel_config`）は計算カーネルの設定のうち `math_fidelity`、`fp32_dest_acc_en`、`dst_full_sync_en`、`bfp8_pack_precise`、`math_approx_mode` しか運ばず、tt-xla の経路では fp32 の CB の読み込みがすべて丸められる。手元の tt-xla の clone（`b360078`、10-02）も同じ
- 影響: tt-xla の経路の LIF は、しきい値のごく近くで CPU と食い違う（`[4·6272, 1024]` の形で 0.005〜0.007%）。カーネルの書き方では直せない。読み込みは切り捨て（round to nearest ではない）で、デバイスの BN の出力に CPU で同じ切り捨てをした LIF をかけると、デバイスの出力とビット単位で一致した（S19）。ただし conv の出力（LIF の入力）がすでに仮数 10 ビットに丸まっているので、SEW-ResNet18 の最初の LIF で D22 が足す反転は 46 個で、conv の丸めによる CPU との差（約 17,200 個）より 2 桁少ない。CPU で D22 だけをまねた網の logits の相関は 0.922 で、ほかの小さな乱れを入れたときと同じ水準（S19）。S18 で「logits の相関が低いのは D22 のため」と推測したのは誤りで、原因は D25 だった
- 扱い: `make_lif_nchw` は SFPU だけで計算する書き方にした（ttnn で直接動かせば正確で、tt-xla が設定を運ぶようになれば正確になる）。tt-xla の新しい版で直っているかは見ていない。GitHub への報告はユーザーが決める

### D23. ベンチマークの推論の段で BN は学習モードだとした（判断）

- 期待（S4 で置いた前提）: SpikingJelly の `benchmark/benchmark_snn_single_gpu.py` は推論でも `model.eval()` を呼ばないので、BN は学習モード（batch の統計で正規化し、running stats を更新する）のまま。p150a でも学習モードにすれば、A30 とそろう
- 実際: 推論の段では `model.train(args.phase == "training")`（手元の 2.0.0 の 537 行目。08-10 の `3b12b104` から）で、モデル全体が推論モードになる。A30 の基準（S4 の batch 8 で 14.8 ms、batch 32 で 51.1 ms）は推論モードの BN で測られ、p150a は S4〜S17 のあいだ学習モードの BN で測っていた。p150a では、学習モードの BN（NCHW で、余白付きのタイルで計算し、前後に `permute` と `reshape` が入り、running stats の更新が 1 層あたり 6 個の小さな演算に分かれる）と前後の並べ替えが、LIF なしの 1 ステップ 122.9 ms のうち約 94 ms を占めていた。推論モードでは、tt-mlir が BN を conv に畳み込み、28.9 ms になる（S18）
- 影響: S4〜S17 の p150a と A30 の比（例えば batch 8 で 25 倍）は、条件のそろわない比較だった。S12 の `a30-hostcost.py`（8.27 ms）は学習モードのまま測っている（`model.eval()` を呼んでいない）
- 扱い: `xla-sew18.py` に `BN=eval` を足した（running stats を入力の batch から写してから推論モードにし、CPU の基準も推論モードにする）。今後の比較は推論モードで行う。A30 の基準も同じ条件で測り直す

### D24. tt-xla 1.4.0 の level 1 は、小さいチャネル数の stem の conv を誤って計算する（実装）

- 期待: optimization_level 1 でも、conv は CPU とほぼ一致する（S10 で level 1 を使い始めたときは、SEW-ResNet18 の logits の相関だけを見ていた）
- 実際: SEW-ResNet18 の stem の conv（3 → 64 チャネル、7×7、stride 2、padding 3、fp32、batch 8 と 32）は、level 1 では CPU との相関が 0.28 前後で、出力の散らばりは CPU の約半分になる。level 0 では 0.999998。入力のチャネルを 4、8、16 にゼロで埋めても誤り（0.21〜0.28）、32 なら正しい（1.000000）。ほかの conv（3×3 と 1×1、64〜512 チャネル）は level 1 でも 0.999997 以上（S18。`tools/xla-conv-check.py`、`tools/xla-stem-pad.py`）
- 影響: S10〜S17 の level 1 の SEW-ResNet18 は、最初の層から誤った値で計算していた。logits の相関（0.85〜0.89）は、logits の大半が線形層のバイアスだったために高く出ていて、誤りを示さなかった。時間は、計算の量が同じなので大きくは変わらない見込み（stem の conv の時間が誤った設定のものだった可能性はある）
- 扱い: `xla-sew18.py` に `STEM_PAD=32`（デバイスのモデルだけ、stem の conv の入力と重みをゼロで 32 チャネルに埋める）を足した。1 ステップは約 4.6〜6.7 ms 増える。tt-xla の新しい版で直っているかは見ていない。GitHub への報告はユーザーが決める
- 追記（S23）: stem を stride folding で 4×4、stride 1 の conv（入力 12 チャネル）に書き換えても同じで、16 チャネルに埋めると CPU との相関 0.39、32 チャネルなら 0.999996

### D25. tt-lang の演算の出力先を `torch.zeros_like` で作れば、呼び出しごとに別のバッファになると思った（実装）

- 期待: `lif_op(x, torch.zeros_like(x))` のように呼べば、呼び出しごとに新しい出力先ができ、結果は呼び出しごとに別のテンソルになる（tt-xla の公式のテスト `tests/torch/ops/test_tt_lang_kernel_e2e.py` も出力先を `torch.zeros_like` で作る。ただし 1 つのプログラムで呼ぶのは 1 回だけ）
- 実際: tt-mlir は形と値が同じ定数を 1 つにまとめる（SEW-ResNet18 の TTNN の IR で、20 個の LIF に対して `ttnn.full` は形ごとの 5 個）。`ttnn.tt_lang_op` は DPS の演算で、TTNN では結果を返さない `ttnn.generic` になり、出力先の引数に直接書き込む。そのため同じ形の LIF がすべて 1 つのバッファに書き、結果はそのバッファから読まれる。順に使い切る間は問題ないが、downsample のあるブロック（layer2.0、layer3.0、layer4.0）の `sn2 + downsample_sn` は `ttnn.add(%5, %5)` になり、後に書いた `downsample_sn` の 2 倍になっていた。layer2 の出力で値 2 の割合が CPU の 0.44% に対して 2.74%、平均が 0.098 に対して 0.145 で、layer4 までで CPU との不一致は `SimpleLIFNode` の 14.4% に対して 28.4% だった（S19）
- 影響: S14〜S18 の tt-lang の LIF を使った SEW-ResNet18 の出力は、layer2 から誤っていた（並べ方によらない。`flat`、`nchw`、`nhwc4` のすべて）。logits の CPU との相関は 0.731 で、元の LIF の 0.908 より低かった。時間には影響しない（演算は同じ）。`xla-ttl-lif.py` のように 1 つの LIF の出力を次の LIF の入力にするだけのつなぎ方では、壊れ方が表に出ない
- 扱い: `xla-sew18.py` の `TtlLIF` は、LIF ごとに別の値で埋めた出力先を渡す（`torch.full_like(x, -1.0 - k)`。カーネルは全要素を書き換えるので、値は結果に影響しない）。IR で `ttnn.full` が 20 個になり、各 `ttnn.generic` が別のバッファに書くことを確かめた。logits の相関は 0.911（`SimpleLIFNode` は 0.908）、1 ステップは 43.5 ms で変わらない。`LIF_OUT=zeros` で元の作り方に戻せる。tt-mlir の側で直すべきもの（DPS の出力先にした定数をまとめない、または `ttnn.generic` の前に出力先を新しく確保する）と見ている。新しい版で直っているかは見ていない。GitHub への報告はユーザーが決める

### D26. A30 と p150a の SEW-ResNet18 を、同じ fp32 の精度で比べているとした（判断）

- 期待: どちらも fp32 のモデルなので、A30 の基準（S4 の 14.8 ms）と p150a の時間は同じ精度の計算どうしの比較になる
- 実際: SpikingJelly の `benchmark/benchmark_snn_single_gpu.py`（502〜503 行目）は `torch.backends.cuda.matmul.allow_tf32 = False` と `torch.backends.cudnn.allow_tf32 = False` を無条件に設定するので、A30 の conv は TF32 を使わない厳密な fp32 で計算している（PyTorch 2.11 の既定は `cudnn.allow_tf32 = True`）。一方 p150a の conv の出力は、仮数 10 ビットに丸まっている（デバイスの stem の BN の出力の 99.84% が仮数 10 ビットに収まる。S19）
- 影響: S18 の「p150a は A30 の約 2.9 倍」は、A30 だけが厳密な fp32 という、p150a に有利な条件での比較だった。TF32 を許すと、A30 は 14.75 ms から 7.5 ms になる（S20）。S12 の `a30-hostcost.py`（8.27 ms）が S4 より速かったのも、主に TF32 のため
- 扱い: `tools/a30-sew18.py` で TF32 を許した時間を測った（S20）。p150a（まとめて同期 31.5 ms）は、TF32 の A30 の 4.2 倍、厳密な fp32 の A30 の 2.1 倍。比較の基準は TF32 の A30 とする（ユーザーの決定、2026-10-28。ベンチマークのままの値は併記する）

### D27. LIF のカーネルは読み書きが律速だとした（判断）

- 期待: LIF は要素ごとの単純な計算なので、メモリの読み書きで決まる。layer1 の 150〜190 GB/s は、1 回に運ぶブロックが 2 タイルと小さいため（S22 の見立て。作業記録の「次にやること」の (b)）
- 実際: layer1 の形で行ごとのサイクル数を測ると（`run-test.sh --auto-profile`）、読み出しのスレッドは 44.8% の時間、バッファの空きを待っていて、計算が律速だった。1 時刻 1 タイルに約 4,500 サイクル。生成された C++ では、tt-lang は `store` ごとに DEST の区間を分け、区間ごとに `init_sfpu`、演算ごとに初期化を呼び、2 つの `store` が同じ部分式を使っても別々に計算する。`floor → relu → sign`（1 行あたり約 30 命令）を 1 時刻に 2 回求めていた
- 影響: ブロックを広げるだけでは足りず、計算を先に減らす必要があった。計算を減らすと、今度は転送（1 コアに 1 件しか転送が飛ばない作り）が前に出た
- 扱い: 閾値を `lt` と `fill` にし、h を運び（`store` が 2 つでも計算し直す量が小さい形）、ブロックを `(rb, bw)` に広げた（S25）。tt-lang のカーネルは、生成された C++（`run-test.sh --emit-runner`）と `--auto-profile` で区間ごとの費用を見てから組み立てる。計算だけ・転送だけの版を作って比べると、どちらが律速かを切り分けられる

### D28. `grid="auto"` は tt-xla の経路でも実機のコアの格子になると思った（実装）

- 期待: `@ttl.operation(grid="auto")` は、コンテナ（ttnn）と同じく、tt-xla の経路でも p150a の計算コアの格子 11×10（110 コア）に解決される
- 実際: tt-xla 1.4.0 の `tt_torch/tt_lang.py` の `_make_deviceless_ttnn_args` は、仮の device（`_StubTtnnDevice(grid_cols=8, grid_rows=8, arch="wormhole_b0")`。注記に「この統合が対象とする 8×8 の Wormhole_b0」）を渡す。tt-lang の `_resolve_grid` は `"auto"` をその `compute_with_storage_grid_size()` で決めるので 8×8 になり、IR の `ttnn.generic` の `core_ranges` も (0,0)–(7,7) だった。`ttl.target_arch` も `wormhole_b0` になる
- 影響: S24、S25 のモデルの LIF は 20 個とも 64 コアで動いていた。コンテナのベンチ（110 コア）の時間は、モデルの中の LIF の時間を表していなかった
- 扱い: `make_lif` と `make_lif4` に `grid` の引数を足し、`xla-sew18.py` と `xla-ttl-lif.py` は `LIF_GRID=11,10` で渡す（S26）。IR の `core_ranges` が (0,0)–(10,9) になり、まとめて同期の 1 ステップが 10.54 → 10.18 ms。`arch` は直していない（橋渡しの中で決まる）

### D29. コンテナで動いたカーネルが、tt-xla の経路の SEW-ResNet18 の中でデバイスを止めた（実装）

- 期待: コンテナ（tt-lang 1.1.10、ttnn）で正しく動いた tt-lang のカーネルは、tt-xla の経路（tt-lang 1.1.5、仮の Wormhole の device で組み立て、D28）でも動く
- 実際: `make_lif4(bf16_out=True)`（fp32 で計算し、スパイクを fp32 の CB から別の区間で bf16 に変えて書く。CB は fp32 が 3 つと bf16 が 1 つ）は、コンテナでは stem の形でも CPU とビット単位で一致した（560 µs）。SEW-ResNet18（`STEM_SPIKE=bf16`、trace）では、グラフのコンパイルの後、最初の実行の途中で止まった（ログが 15:49:28 から進まず、1 本のスレッドが CPU を回し続けた）。SIGINT では止まらず、SIGTERM で止めた後、コンテナで小さい LIF を流しても止まったので、`tt-smi -r` で戻した（2026-10-28 16:47）
- 分かっていること: IR は狙いどおり（LIF が bf16 の `%arg67` に書き、`max_pool2d` が直接読む。maxpool の前の `typecast` は消えた）。tt-xla が生成した計算のカーネルは、ブロックの大きさを除けばコンテナのものと同じ構成で、区間ごとに `init_sfpu(入力, 出力)` でパッカーを設定し直していた。LIF のカーネル、モデルの中での L1 の使い方（CB は 1 コア約 850 KB）、maxpool のどれで止まったかは切り分けていない
- 扱い: `STEM_SPIKE=bf16` は使わない。切り分けるには、bf16 の `make_lif4` を tt-xla の経路で単体で流す試験と、`PREFIX=sn1` で maxpool の前まで流す試験が要る（どちらも止まればリセットが要る）

### D30. tt-xla から conv の設定や L1 への置き方を渡せない（実装）

- 期待: S28 で、tt-xla 1.4.0 のプラグイン（`pjrt_plugin_tt.so`）の文字列に、tt-mlir の `override-conv2d-config`、`override-output-layout`、`insert-memreconfig` が見えた。コンパイルのオプションから、これらか、L1 に置く狭い指定を渡せるかもしれない
- 実際（S29。サブエージェントが手元の tt-xla の clone（`b360078`。1.4.0 は `941a6b8` のビルドで、clone の方が新しい）と 1.4.0 の `.so` を読んだ。実機では試していない）: コンパイルのオプションは `pjrt_implementation/src/api/compile_options.cc:16-142` の決まったキー（`optimization_level`、`enable_trace`、`math_fidelity`、`fp32_dest_acc_en`、`experimental_weight_dtype` など）だけで、知らないキーは黙って捨てる（`compile_options_parser.cc:349-351`）。tt-mlir に渡すのは `module_builder.cc:1016` の `optimizationLevel` などの決まった項目だけで、パイプラインのオプションを素通しする口はない。`.so` にある名前は tt-mlir の定義がリンクされているだけ。level 0、1 は活性化を全部 DRAM に置き、L1 に置くのは level 2（`docs/source/performance.md:49-68`。conv を含むとコンパイルできない。D9、D15）。`experimental_weight_dtype` は matmul と linear だけで conv には効かない（`performance.md:160`）
- 分かっていること: コンテナの ttnn で conv の入出力を L1 に置き、`enable_act_double_buffer` を使うと、stem を除く conv は 3.71 → 2.34〜2.74 ms、LIF（layer1〜4）は 1.27 → 0.92 ms になる（S29、単体の和）。tt-mlir が付ける設定（`act_block_h_override`、シャードの指定なし、double buffer なし、DRAM の interleaved）のままでは、この分は取れない
- 追記（S29、網全体）: IR をつないで流し直す `ttnn-replay-chain.py` で、layer1 以降の活性化を L1 に置き、conv に double buffer を使い、入らないテンソルだけを DRAM に戻すと、1 ステップのデバイスの稼働は 9.84 → 7.97 ms になった（コンテナ。LIF のスパイクは全層で CPU と一致）。tt-xla の経路で取れていない分は、少なくともこの 1.9 ms
- 扱い: tt-xla の経路の中では、conv と活性化の置き場所は詰められない。回り道の候補は `backend=codegen_py`（TTNN の Python のコードを書き出し、手で直して `codegen_load_py` で読み込む。`compile_options.h:12-31`、`compile_options.cc:54-66`。p150a、trace、tt-lang の演算で動くかは確かめていない）か、モデルを ttnn で直接書くこと（公式の ResNet-50 と同じ形）

### D31. 1 ステップごとに同期するときの上乗せを、tt-xla の実行時の費用だとした（判断）

- 前提: SEW-ResNet18 で、1 ステップごとに同期（S30 で 14.2 ms）がまとめて同期（9.59 ms）より約 5 ms 長いのは、trace の起動、出力の読み戻し、ホストの送り出しが、デバイスの計算と重ならないため（S24 の「次にやること」の (c)）。回ごとに 1〜2 ms ぶれる理由は分かっていなかった
- 実際（S31）: 入力の大きさによらず、ごく小さいグラフでも、プロセスごとに 0.4 ms の回と 4 ms の回に分かれた。遅い回はスレッドの実際の周波数が 800 MHz（`acpi-cpufreq` の最低）で、Python の呼び出しまで遅い。SMT の兄弟（156〜167）に SCHED_IDLE の空回りを置き、計測を 60〜71 に固定すると、小さいグラフは 0.21 ms、SEW-ResNet18 は 10.99 ms（ぶれもほぼ消えた）。待つスレッドが眠る（または `yield` する）ので、`schedutil` から見た使用率が低く、周波数が上がらず、アイドルからの復帰の遅れも毎回入ると見ている（C ステートは見ていない）。A30 の CUDA の同期は空回りで待つので、ホストの CPU は起きたままになる（推測）
- 教訓: ホストとデバイスの往復を含む時間がプロセスごとに二峰に分かれたら、デバイスより先に、ホストの CPU の周波数（`thread-sampler.py` の APERF/MPERF の値）を見る
- 扱い: 表の値は既定の省電力のまま測っている。`performance` の governor（sudo）か、計測のときの空回りを採るかは、ユーザーが決める。採るなら A30 も同じ条件で測り直す

### D32. tt-xla に、入力をタイルの形で渡せるかもしれないとした（実装）

- 前提: stem の入力は毎ステップ、デバイスの上で行優先からタイルに直している（2 つで 0.86 ms。幅が 32 しかないので遅い）。ホストでタイルにしてから渡せば消せるが、比べる条件の判断が要る（案の一覧の x3）
- 実際（S33）: 別にコンパイルしたグラフの出力（デバイスの DRAM のタイル）を次のグラフに渡しても、次のグラフの trace の入力は `#system_memory` の行優先の型のままで、trace の中で `to_layout` をする。時間もホストの入力を直接渡す場合と同じだった。tt-xla 1.4.0 には trace の入力の型を変えるオプションもない（D30）
- 扱い: tt-xla の経路では、入力のタイル化は消せない。ttnn で直接書けば、入力を初めからタイルの形で置ける（案の一覧の t4）
