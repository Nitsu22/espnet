# `run_npz_triplet_rand_sch_noise.sh` 実行時の挙動まとめ

対象スクリプト: `egs2/whamr/se_npz/run_npz_triplet_rand_sch_noise.sh`

このスクリプトは、**WHAMR!（2-speaker / 2ch / 残響あり）相当のNPZ/Numpyデータ**を使って、
**空間埋め込み（spatial embedding）をTriplet lossで学習**します。
出力は「分離・強調音」ではなく、**1サンプル→256次元程度の埋め込み**を作るためのモデルです。

---

## 1. 何が実行されるか（結論）

- `enh_npz.sh` の **Stage 6（学習）だけ**を実行します（`--stage 6 --stop_stage 6`）。
- 学習データは `tr_mix_clean_reverb_min_8k`、検証は `cv_mix_clean_reverb_min_8k` を使用します（`min_or_max=min`, `sample_rate=8k`）。
- 学習設定は `conf/tuning/train_se_mc_conformer_div_npz_triplet_rand_scheduler_noise.yaml` を使用します。
- Lossは **Triplet loss**、LRスケジューラは **ReduceLROnPlateau**、negativeは **毎呼び出しランダム**に生成します。

---

## 2. 実行コマンド（実質）

`run_npz_triplet_rand_sch_noise.sh` は内部で以下を呼びます（`"$@"` で上書き可能）:

```bash
./enh_npz.sh \
  --train_set tr_mix_clean_reverb_min_8k \
  --valid_set cv_mix_clean_reverb_min_8k \
  --test_sets "tt_mix_clean_reverb_min_8k" \
  --fs 8k \
  --ngpu 1 \
  --ref_num 2 \
  --enh_config ./conf/tuning/train_se_mc_conformer_div_npz_triplet_rand_scheduler_noise.yaml \
  --enh_exp exp/enh_train_se_mc_conformer_div_triplet_rand_scheduler_noise \
  --stage 6 \
  --stop_stage 6
```

補足:
- `--test_sets` は Stage 6 では基本的に使われませんが、`enh_npz.sh` の必須引数なので指定されています。
- `--use_noise_ref false` は「教師信号としてnoiseを別途与えるか」のフラグで、このタスク（contrastive）では基本的に使いません。

---

## 3. 入力データ（どんなデータを読むか）

Stage 6 では `data/${train_set}/npz.scp` をキーにして、以下のファイル群を読みます（`enh_npz.sh` 内の指定に基づく）。

例: `data/tr_mix_clean_reverb_min_8k/`

- `npz.scp` → `npz_path`（メタ情報: sample_rate, split, wham_*_scale, start_samp_16k など）
- `spk1_base_npz.scp` → `s1_base`（話者1ベース波形, `.npy`）
- `spk2_base_npz.scp` → `s2_base`
- `spk1_temp_npz.scp` → `s1_temp`（ルームシミュレーション用のテンプレ波形）
- `spk2_temp_npz.scp` → `s2_temp`
- `noise_base_npz.scp` → `noise_base`（WHAMノイズ波形, `.npy`）
- `rir_npz.scp` → `rir_path`（RIR, `.npz`）
- `room_param_npz.scp` → `room_param_path`（部屋寸法/マイク/話者座標/T60 等, `.npz`）

※ `mix_*` の見た目に関係なく、**実際にnoiseを混ぜるかは後述の `mix_type`（前処理設定）で決まります**。

---

## 4. 前処理（NPZから何を作るか）

前処理は `preprocessor: npz`（`espnet2/train/preprocessor_npz.py:NpzPreprocessor`）で行われます。
この設定（noise版）の主なポイント:

- `mix_type: both`
  - **入力波形 = speech1 + speech2 + noise**（noiseあり）
- `ref_condition: reverb`
  - 参照する音声は **reverberant（残響あり）**側
- `speech_segment: 32000`
  - 学習時に **約4秒@8kHz**へランダム切り出し（`avoid_allzero_segment: true`）
- `output_audio_subtype: PCM_16`
  - 量子化（PCM16相当）を挟む
- `contrastive_enable: true` で、1サンプルから **Anchor / Positive / Negative** を生成

### Anchor / Positive / Negative の作り方（Triplet用）

- **Anchor**: 元のroom/RIRで生成した波形（必要に応じてcrop）
  - `anchor_single_channel: true` のため、**Anchorだけ 1ch（ch0）**に落とす
- **Positive**: Anchorと**同じroom/RIR**のまま、speech1/speech2のスケールをランダムに変えて生成
  - `contrastive_scale_min/max: 0.9–1.1`（話者ごとに独立）
- **Negative**: 別サンプルから引いた**別room/RIR**に差し替えて生成（speech/noise素材は同じ）
  - negativeは `contrastive_num_neg: 1`（1個）
  - `contrastive_random_each_call: true` のため **毎回ランダムにnegativeを引く**
  - negative候補は `rir_npz.scp` と `room_param_npz.scp` から構成されるプール
    - `enh_npz.sh` が `contrastive_pool_*_scp.{tr,cv}` を自動指定（splitごとにプールを分ける）

---

## 5. 学習モデル（何を学習しているか）

学習タスク: `espnet2.bin.enh_se_npz_train`（`espnet2/tasks/enh_se_npz.py:SpatialEncoderTask`）

### 入力と出力

- 入力: `speech_anchor`（1ch）と `speech_pos/speech_neg*`（2ch）
- 出力: 各入力から **L2正規化された埋め込みベクトル**（`embed_dim: 256`）

### 構成

- Encoder: STFT
  - `n_fft: 256`, `hop_length: 64`（onesidedで `freq_bins: 129`）
- Spatial encoder: `mc_conformer_div`（`MCConformerDivSpatialEncoder`）
  - **SC用とMC用で別パラメータ**（SC branch / MC branch）
  - lossで **SC埋め込み（Anchor）をMC埋め込み（Positive）へ近づける**ように学習

---

## 6. 損失（Triplet）と最適化

Loss設定（`train_se_mc_conformer_div_npz_triplet_rand_scheduler_noise.yaml`）:

- Triplet loss（`margin: 0.2`）
  - `max(0, d(anchor,pos) - d(anchor,neg) + margin)`
  - 埋め込みはL2正規化して距離計算

最適化/スケジューラ:

- Optimizer: Adam（`lr: 1e-4`）
- Scheduler: ReduceLROnPlateau（`mode: min`, `factor: 0.1`, `patience: 10`）
- 早期終了: `patience: 20`（`valid.loss` が改善しない場合）

---

## 7. 生成される成果物（どこに何が出るか）

出力先: `egs2/whamr/se_npz/exp/enh_train_se_mc_conformer_div_triplet_rand_scheduler_noise/`

典型的に生成されるもの:

- `train.log`（学習ログ）
- `config.yaml`（実際に使われた設定）
- `checkpoint.pth`（resume用）
- `latest.pth`（最新エポックへのsymlink）
- `valid.loss.best.pth`（検証loss最良へのsymlink）
- `Nepoch.pth`（エポックごとのスナップショット）
- `run.sh`（同じ条件で再実行/再開するためのスクリプト）

※ `enh_npz.sh` は内部で `--resume true` を渡すため、同じ `--enh_exp` で再実行すると基本的に**続きから再開**します。

---

## 8. 何が学習できると期待できるか

この設定は「音声分離/強調」そのものではなく、**空間（部屋/RIR）に関する埋め込み表現**の学習です。
特に、

- **同一room/RIR**（Anchor↔Positive）は近い埋め込み
- **別room/RIR**（Anchor↔Negative）はmargin以上に離れる埋め込み

となるよう最適化されます。

加えて、Anchorは **1ch**、Positive/Negativeは **2ch**なので、
**「1chでも、2ch由来の空間表現に一致する埋め込みを出す」**ことを狙った学習になっています。
結果として、以下のような性質が期待できます:

- 話者内容やゲイン変動（0.9–1.1）に対して頑健な「部屋っぽさ」の表現
- 近傍探索やクラスタリングで、似た残響/配置のサンプルが近くなる
- 下流タスク（例: 別モデルの条件付け特徴など）に流用できる可能性

---

## 9. 実行上の注意（詰まりやすい点）

- 前処理で `pyroomacoustics` と `scipy` を使うため、環境に依存して**データローダが重い**です（CPU負荷/IOに注意）。
- Stage 6 のみ実行するため、初回は `exp/enh_stats_8k/`（shape等）が無いと困ることがあります。
  - 必要なら `./run_npz_triplet_rand_sch_noise.sh --stage 5 --stop_stage 6` で stats→学習まで実行します。
- 学習条件を変えて「完全に最初から」やり直す場合は、`--enh_exp` を変えるか該当ディレクトリを退避/削除してください（resumeされるため）。

