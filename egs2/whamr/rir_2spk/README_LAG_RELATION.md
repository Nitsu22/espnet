# 二話者Rec-RIR：遅延別の関係抽出と時間平均

## 実装範囲と状態

`rec_rir_pit_lag_relation` は、Rec-RIRの音声側を維持し、CTF推定部を
`[q; k; q×k] → 関係MLP → 時間平均 → 遅延処理 → CTF出力` に置き換える。
NB版とFull版を用意する。話者別投影は追加せず、最後のLinearで二話者分を出す。

この変更では実装・設定・説明のみを追加した。テストの作成・実行、モデルの
forward/backward、学習、推論、評価、GPUメモリ計測は実施していない。
動作と性能は未検証であり、既存Baselineとの公平な比較条件も実験開始前に確認する。

## 形状と処理

既定値は `S=2, F=129, D=96, N=60`。Bはバッチ、Tは有効な音声フレーム数、
RIは実部・虚部の2成分を表す。

| 処理 | 形状 |
| --- | --- |
| Clean／Reverb共通特徴 | 各 `[B,F,T,D]` |
| 独立したLinearでq／kを作る | 各 `[B,F,T,D]` |
| 遅延dでqの時刻t、kの時刻t+dを連結 | `[B,F,T-d,3D]` |
| `Linear(3D,D) → LeakyReLU → Linear(D,D)` | `[B,F,T-d,D]` |
| 有効なT-dフレームを平均 | `[B,F,D]` |
| d=0…N-1をStack | `[B,F,N,D]` |
| CTF専用ブロック6層 | `[B,F,N,D]` |
| `Linear(D,D) → LeakyReLU → Linear(D,4)` | `[B,F,N,4]` |
| 話者と実虚を分け、軸を並べ替える | `[B,S,RI,F,N]` |

4出力の順序は `話者1実部、話者1虚部、話者2実部、話者2虚部`。
ネットワーク出力は因果順のH[0], H[1], …, H[N-1]で、Nは音声の時間軸Tとは
別の遅延軸。`return_embedding=True` はCTFブロック後の `[B,F,N,D]` を返す。

- **NB版**：既存の `SpatialNetLayerNB` を使い、双方向MambaがN方向を処理する。
- **Full版**：既存の `SpatialNetLayer` を使い、周波数F方向と遅延N方向を処理する。
  CTF側のFull-band重みは音声側から独立。`full_share=0` ではCTFスタック内の
  Full-band Linearを層間共有する。全層の全パラメータを共有する意味ではない。

音声Decoderは `[B,S,RI,F,T]` を返す。3ロスの重みは各1で、同じ話者割り当てを
Clean・Reverb・CTF再構成に使う。再構成に用いる音声は教師のdirect音声である。
既存の `estimate_ctf()` による一度だけのタップ反転と、ordinary sweep PIMを継承する。

## 可変長とメモリ

混合と4つの教師音声の有効長から、発話ごとの共通長を取る。同じ共通長の発話を
まとめ、波形をその長さに切ってから既存wrapperのSTFT・ネットワーク・PITロスへ渡す。
平均だけのマスクでは双方向特徴抽出器へパディングが入るため、特徴抽出前に長さを揃える。
各グループの発話平均ロス・統計を、グループ内の発話数で重み付けしてバッチ平均へ戻す。
推論も実長ごとに処理し、最後に元のバッチ順へ戻す。

長さを省略したテンソルは全長を有効と扱う。パディングがある場合は対応する
`*_lengths` を渡す。長さは正の整数かつ保存された波形幅以下でなければならない。
`T<N` の音声は、実長と必要長を示してエラーにする。既定の中心化STFT
（n_fft=256、hop=128）とN=60では、最低7552サンプルが必要。
この下限は推定精度を保証する長さではない。STFTの反射パディングに必要な下限も確認する。

q／kは一度だけ計算し、遅延ごとのスライスを順次処理する。時間平均はFP32で集計して
特徴dtypeへ戻す。`relation_checkpoint: true` の学習時は、要素積・連結・MLP・平均を
一つの関数にして `checkpoint(..., use_reentrant=False)` を使う。関数への入力は
その遅延のスライスを明示的に渡し、戻り値は `[B,F,D]` のみにする。
推論時・勾配無効時はCheckpointingしない。メモリと再計算時間の効果は未計測。

## 設定と使い方

追加設定：

- `conf/tuning/train_rec_rir_2spk_nf_8k_lag_relation_nb.yaml`
- `conf/tuning/train_rec_rir_2spk_nf_8k_lag_relation_full.yaml`

モデル固有項目以外は `train_rec_rir_2spk_pit_clean_8192.yaml` と同じ。
8 kHz、STFT 256/hop128/sqrt-Hann、4秒crop、音声6層・ノイズ2層・CTF6層、
AdamWと既存のscheduler・モデル選択条件を引き継ぐ。パディング除外のため、
可変長バッチのロス処理は旧Baselineと異なる。比較時は入力長・バッチ設定も確認する。

```yaml
rir_model_type: rec_rir_pit_lag_relation
preprocessor: rec_rir_pit
model_conf:
    ctf_length: 60
    ctf_block_type: nb  # Full版はfull
    relation_checkpoint: true
```

上記は新しい項目の抜粋。実際には追加した設定ファイル全体を使う。
新モデルでは `dim_output_CTF` を指定しない。タップ数は `ctf_length`、各タップの
出力幅は常に `num_spk * 2 = 4` として別管理する。

既存の `rir.sh` では、設定ファイルに加えて
`--rir_model_type rec_rir_pit_lag_relation` を明示する。二話者用の
`speech_direct1/2.scp` と `speech_reverb1/2.scp` を渡す経路へ接続済み。
入力だけでバッチサイズを決める場合の `local/prepare_input_shapes.py` も新モデルを受け付ける。
NF-WHAMR用の既存8 kHz dumpを使い、新しいstats・学習出力先を指定する。

推論は既存の `espnet2.bin.rec_rir_pit_inference` を利用できる。
保存された新モデルのconfig・checkpointを指定し、`--sample_rate 8000` を使う。
既存LS評価と揃える場合は `--rir_length 16000 --output_subtype FLOAT` を明示する。
Baselineのcheckpointを新Headへそのまま読み込む互換性はない。

## 後続の検証項目（今回は未実施）

- NB／Full両設定からの構築、学習・推論出力形状、state_dict保存・再読み込み。
- 既知の遅延を用いたq[t]/k[t+d]の対応、因果タップ順、PIMでの反転回数。
- Nと同じフレーム数、N未満、パディング値を変えた可変長バッチ、推論の順序復元。
- 長さグループの発話数が異なる場合と、一発話ずつ処理した場合の損失・統計の一致。
- 関係MLP・投影・音声側・CTF側への勾配、有限値、AMP時の安定性。
- Checkpointingの有効／無効での出力・勾配一致、ピークメモリと速度。
- 基底wrapperのfactory追加による、既存BaselineとCTFSplit等の構築・checkpoint互換性。
- CTF側と音声側のパラメータ独立性、Full版のスタック内共有、3ロスの共通PIT。
