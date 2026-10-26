# BiLSTM CTF構造・損失の比較設定（8 kHz）

基準は `train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4_time_l1_plateau_200ep.yaml`。
2話者・雑音なし、時間BiLSTM 8段、Pooling後の周波数BiLSTM 4段、特徴幅64、CTF 60 taps。
音声推定・音声再構成の枝は追加しない。以下は実装・検証用の設定であり、学習済み結果ではない。

## 独立した比較

| variant | 基準からの変更 | 損失 |
| --- | --- | --- |
| `baseline` | 変更なし | Sweepスペクトル＋時間Sweep L1 |
| `pre_last2_bilstm` | 前段の周波数処理だけを、最初6段はlight、最後2段はBiLSTMにする | 同上 |
| `rank32` | light周波数MLPのrankだけ16→32。特徴bottleneckは16のまま | 同上 |
| `speaker_query` | 発話に応じた話者queryで時間Pooling重みを1回再計算する | 同上 |
| `rir_mse` | 構造は基準と同じ。時間Sweep L1を逆フィルタ後の波形MSEに置換 | Sweepスペクトル＋RIR MSE |

さらに `pre_last2_bilstm_rir_mse`、`rank32_rir_mse`、`speaker_query_rir_mse` を用意した。
構造の効果は同じ損失どうし、損失の効果は同じ構造どうしで比較する。
このランチャーは指定した1条件だけを実行し、一括投入は行わない。

全条件でseed=0、スクラッチ、LR=0.001、5 epoch停滞でLR半減、最大200 epoch、
15 epoch停滞で早期終了を維持する。前段最後2段の置換・query追加では、同じseedの基準モデルと
共通パラメータの初期値も一致する。rank32は既存のrank指定を使うため、パラメータ形状の変更に
伴って後続の乱数消費も変わり、全共通初期値の一致は保証しない。

## 実装上の定義

### 前段の混在

`model_conf.predictor_conf.pre_frequency_modules` に前段数と同じ長さのリストを指定する。
例：`[light, light, light, light, light, light, bilstm, bilstm]`。
従来の `pre_frequency_module` は引き続き全段共通の指定として使える。
リスト指定時は共通指定を既定の `light` にする。時間BiLSTM 8段と後段4段は変更しない。
周波数BiLSTMは各観測フレームに適用するため、後段だけを増やす場合より計算量が増える。

### 発話条件付き話者Pooling

`model_conf.predictor_conf.pooling_mode: speaker_query` で有効化する。
まず従来の話者別・周波数別・チャネル別時間Poolingを行う。その出力を周波数方向に平均し、
話者slot埋め込みを加え、LayerNormと線形変換で発話依存の2つのqueryを作る。
元の各時刻・周波数の特徴から作ったkeyとの内積を、元のPooling logitsへ加える。
時間方向のsoftmaxを取り直し、元の特徴をもう一度集約する。

queryは全周波数で共有するが、keyと時間重みは周波数ごとに異なる。refinementは1回。
時間周波数点を排他的にどちらかの話者へ割り当てたり、音声を復元したりする処理ではない。
既存の可変長・パディングの扱いを維持し、全バッチ時間軸にsoftmaxをかける。
既定の `pooling_mode: legacy` は従来のstate dictと演算経路を保持する。

### 逆フィルタ後の教師ピーク正規化MSE

`model_conf.rir_mse_loss_weight` が正のときだけ有効。既定値は0。
学習では従来どおり、正解の直接音RIRを通したSweepを推定CTFへ入力する。
推定応答と正解残響応答の両方を、同じSTFT/iSTFT・元のSweep逆フィルタ・
既知の測定利得と遅延補正を通して2秒の波形に戻す。
逆変換後の推定を `p`、教師を `r` とすると、話者・サンプルごとの損失は

```text
scale = max(max(abs(r)), rir_mse_loss_eps)
loss_rir_mse = mean(((p - r) / scale) ** 2)
loss = loss_sweep + rir_mse_loss_weight * loss_rir_mse
```

`rir_mse_loss_eps` は振幅の下限（既定1e-8）。推定側の独立ピーク正規化、
ピーク位置合わせ、利得当てはめは行わない。利得・遅延・極性の誤差を残す。
教師を生の物理RIRやLS参照に直接置き換えず、両側の測定処理を一致させる。
2秒外の余分な応答にも損失を与えるため、完全長のSweepスペクトル損失を維持する。

追加設定では時間L1を0、MSE重みを10,000とした。seed=0の基準構造の初期値に対し、
学習データの先頭3混合音（各4秒）で同じスペクトルPIT割当てを使い、CTF出力に対する
勾配のノルムを比較した。スペクトル損失/MSEの勾配比は12,006、15,128、12,821だった。
重み1.0ではMSEの影響が極端に小さいため、同程度の勾配規模になる10,000を初期設定にした。
**これは3例・初期値だけの尺度合わせであり、最適重みや学習中の勾配均衡は保証しない。**
学習ログと検証データで調整する。テストデータは重み選択に使っていない。
各損失の合計で1つの話者PITを選び、その共通割当てで成分統計も集計する。
既存の時間L1・DRR補助損失と併用することもできる。

## モデル選択と評価

今回の比較設定では構造・損失以外の変更を混ぜないため、LR調整・早期終了・最良モデル選択を
基準と同じ `valid.loss_sweep_min` に固定している。検証RIR RMSEによるモデル選択への変更は
今回の実装に含めていない。新しい `loss_rir_mse` は合計損失で選んだPITの値、
`loss_rir_mse_min` はMSE単独でPITを最小化した診断値。後者はテストのLS RMSEではない。

通常PIMによる推論は引き続き直接音→残響の相対フィルタを出力する。
この損失だけで物理clean→reverb RIRへ出力の意味が変わるわけではない。
LS参照による形状評価と物理WHAMR参照による評価は分けて解釈する。
RT60評価器の不適切な回帰区間採用など、評価処理の修正も今回の変更に含めていない。

## 起動

設定だけを確認する場合（学習・ファイル作成なし）：

```bash
bash run_pooled_bilstm_ctf_ablations_8k.sh pre_last2_bilstm --print-config
bash run_pooled_bilstm_ctf_ablations_8k.sh rank32 --print-config
bash run_pooled_bilstm_ctf_ablations_8k.sh speaker_query --print-config
bash run_pooled_bilstm_ctf_ablations_8k.sh rir_mse --print-config
```

学習時は実行先のGPUスキル手順に従い、既存の8 kHz dumpと専用の出力先を指定する。
例（この実装作業では未投入）：

```bash
bash run_pooled_bilstm_ctf_ablations_8k.sh pre_last2_bilstm \
  --stage 5 --stop_stage 6 \
  --dumpdir /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_nf_2spk_8k_min \
  --rir_exp exp/rir_bilstm_pre_last2_bilstm_8k
```

ランチャーは既存チェックポイントの上書き防止を元のレシピに委ね、variantごとに
異なる既定タグを使う。実行用キャッシュは許可済みのcheckout内の専用一時ディレクトリに置き、
終了時に削除する。新しいモデルオプションは設定ファイルに保存されるため、推論にはこの実装を
含むコードと対応する設定を使う。

## 検証

`test/espnet2/rir/rec_rir/test_bilstm_ctf_ablations.py` は、設定差分・ランチャーの対応、
周波数処理の軸、queryの発話依存性、教師正規化の利得・極性・遅延の扱い、
同じ逆フィルタ処理、話者PITと勾配、独立・組合せ7条件の保存／再読み込み／2話者RIR推論を確認する。
CPUで実際のBiLSTMを使用し、Mambaの構築は不要。
既存のPooling・周波数処理・Sweep L1・DRRの回帰テストも合わせて実行する。

実装時は関連8モジュールの38テスト中37件成功、CUDA依存の既存Mambaテスト1件をスキップ。
全8条件について実設定（8 kHz、129周波数、60 taps、2秒RIR、入力音声は1024サンプル）での
CPU順伝播・逆伝播も確認した。ただし、この実サイズ確認時のMSE重みは1.0。
重み10,000への変更後は上記38テストを再実行した。実サイズの再確認コマンドは自動承認の
タイムアウトで実行できなかった。GPUでの学習・性能評価は未実施。

| 構造（損失変更によるパラメータ増加なし） | パラメータ数 |
| --- | ---: |
| 基準 | 1,090,400 |
| 前段最後2段の周波数BiLSTM | 1,277,790 |
| rank32 | 1,123,552 |
| 話者query Pooling | 1,098,848 |
