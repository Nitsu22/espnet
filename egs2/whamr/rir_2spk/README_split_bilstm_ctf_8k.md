# CTF推定: SR-CorrNet参考

Linear: https://linear.app/nitsu22/issue/DAI-106/ctf推定-sr-corrnet参考

SR-CorrNetの固定話者Splitと話者間相互作用を参考にした、2話者CTF推定モデル。
入力は従来の混合音STFT、出力は話者別CTF。音声再構成の枝は追加しない。

```text
Conv encoder → [時間BiLSTM → 軽量周波数処理] × 4
             → 固定2話者MLP Split（各話者64次元）
             → [共有時間BiLSTM → 共有軽量周波数処理 → 話者Attention＋FFN] × 4
             → 話者ごとの共有時間Pooling
             → 共有周波数BiLSTM × 4 → 共有CTF head
```

## 仕様

- 前段の時間BiLSTMは合計8個。後半4個は2話者の枝へ同じ重みを適用する。
- 特徴幅64、周波数rank16、bottleneck16、後段周波数BiLSTM4個を維持。
- Splitは各TF点の `64 → 512 → 2*64` MLP（SiLU）とLayerNorm。
  全幅の話者特徴2本を学習し、チャネルの単純な半割や同一特徴の複製は行わない。
- 後半の各段は `時間 → 周波数 → 話者` の順序。話者Attentionは各TF点で長さ2の
  Self-Attention（4 heads、pre-LayerNorm、残差）、続いて `64 → 256 → 64` の
  ReLU FFN（pre-LayerNorm、残差）。話者位置エンコーディングは使わない。
- Poolingは各話者・周波数・チャネルで時間softmaxを取る。スコアネットは
  `64 → 32 → 64`（GELU）で話者間共有。既存の2話者生成Poolingとslot埋め込みを置換する。
- Split後の時間・周波数処理、Pooling、CTF headはすべて話者間で重み共有。
  Split出力以降は、枝を交換するとCTFも対応して交換される。
- 従来の可変長・paddingの扱いを保ち、Poolingは全バッチ時間軸へsoftmaxをかける。
  最終の発話単位PITは既存実装を使う。

パラメータ共有でも後半は2話者分を処理するため、前段TF処理の実行量は
従来8段に対して概ね `(4 + 2*4)/8 = 1.5` 倍となり、Attentionも追加される。
パラメータ数は **1,387,168**（比較元1,090,400から約27%増）。
実時間とGPUメモリはGPU上の検証で確認する。

## 損失と学習

`loss = loss_sweep + 10000 * loss_rir_mse`。時間Sweep L1は0。
MSEは推定応答と教師応答を同じSTFT/iSTFT・較正済みSweep逆フィルタへ通した後、
教師のピーク振幅だけで両側を正規化する。独立した推定側のピーク正規化や位置合わせはしない。
2秒外も拘束するため完全長のスペクトル損失を維持し、合計損失で共通の話者PITを選ぶ。
通常PIMの推論が返す相対CTF/RIRの意味は従来と同じ。

重み10,000は既存の④（MSEのみ変更）とそろえた初期値で、新構造での最適値は未確認。
seed0、スクラッチ、LR0.001、5 epoch停滞で半減、最大200 epoch、15 epoch停滞で早期終了。
LR・早期終了・最良モデル選択は既存比較と同じ `valid.loss_sweep_min` を使う。

設定の `model_conf.predictor_type: split_bilstm` で新predictorを選択する。
既定値 `pooled` は従来predictorの構築とチェックポイント形式を維持する。

## 起動

レシピから設定だけを確認する（学習・ファイル作成なし）：

```bash
bash run_split_bilstm_2spk_nf_8k_sweep_v2.sh --print-config
```

学習時は実行先のGPUスキル手順に従い、GPUを明示し、新しい出力先を指定する。
以下は起動例であり、この実装作業で投入したジョブではない。

```bash
bash run_split_bilstm_2spk_nf_8k_sweep_v2.sh \
  --stage 5 --stop_stage 6 --ngpu 1 \
  --python /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python \
  --dumpdir /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_nf_2spk_8k_min \
  --rir_exp exp/rir_split_bilstm_enc4_dec4_post4_mse_8k
```

実行前に既存の `local/check_two_speaker_training.py` で実データ・batch4・最長検証入力・
PIT・逆伝播・2話者RIR推論を確認する。今回のコードは実行中の4条件と別のworktreeに置く。

ラボ用の `local/run_split_bilstm_8k_lab.sh GPU_INDEX RESULT_RECIPE` は、GPUの空きを確認し、
上記のGPU検証に成功した場合だけスクラッチ学習を開始する。専用結果名には実行ホストを含め、
既存結果を上書きせず、開始情報・検証結果・終了状態・最良チェックポイントの完了記録を保存する。

## 実装時の検証

- 新規7テスト成功：Split、話者Attentionの局所性・他話者への依存、LSTMの処理軸、
  Split後全体の話者交換同変性、共通PIT、全パラメータの有限勾配、保存・再読み込み・2話者RIR推論。
- 既存関連38テスト中37件成功。CUDA依存の既存Mambaテスト1件はCPU検証のためスキップ。
- 実設定（129周波数、60 taps、2秒RIR、MSE重み10,000）のCPU順伝播・逆伝播を確認。
  この確認のみ入力音声は合成1024サンプル・batch1。CTF出力は `[2,129,60]`。
- GPUでのbatch4検証、本学習、性能評価は未実施。

## 参考資料

- 論文: https://arxiv.org/html/2603.29097v2 （III-B〜III-D）
- 公式実装: https://github.com/dmlguq456/SR_CorrNet_SS

固定SplitのMLPは公式実装のSiLU構成を参考にし、話者Attention＋FFNは原論文の
位置エンコーディングなしTransformer構成を参考にした。Correlation入力・可変話者数Attractor・
音声フィルタ出力・途中の音声教師は導入しておらず、SR-CorrNet全体の再現ではない。
