# se_npz 学習設定まとめ（実装確認済み）

このドキュメントは `egs2/whamr/se_npz` の **現行実装**を確認したうえで、
学習設定と「何が学習されるか」の想定を整理したものです。

## 1. 対象構成（実際に読んだファイル）

- 学習スクリプト: `egs2/whamr/se_npz/enh_npz.sh`
- 実行ラッパ: `egs2/whamr/se_npz/run_npz.sh`
- タスク定義: `espnet2/tasks/enh_se_npz.py`
- モデル実装: `espnet2/enh_se/espnet_model_npz.py`
- 前処理: `espnet2/train/preprocessor_npz.py`
- Triplet Loss 実装: `espnet2/enh_se/loss/criterions/contrastive_loss.py`
- Triplet Wrapper 実装: `espnet2/enh_se/loss/wrappers/contrastive_loss_wrapper.py`
- 損失・設定: `egs2/whamr/se_npz/conf/tuning/train_se_mc_conformer_div_npz_triplet.yaml`

## 2. 入力データと前処理

### 入力
`data/*_mix_clean_reverb_min_8k/` の以下を使用:

- `npz.scp` → `npz_path`
- `spk1_base_npz.scp` → `s1_base`
- `spk2_base_npz.scp` → `s2_base`
- `spk1_temp_npz.scp` → `s1_temp`
- `spk2_temp_npz.scp` → `s2_temp`
- `noise_base_npz.scp` → `noise_base`
- `rir_npz.scp` → `rir_path`
- `room_param_npz.scp` → `room_param_path`

### 前処理の実動作（NpzPreprocessor）
- **WHAMR! と同じ生成**になるよう、`pyroomacoustics` の `simulate()` を使用
- `room.generate_audio()` を **anechoic / reverberant の2回**呼び出す
- `ref_condition: reverb` のため **reverberant を参照**に使う
- `mix_type: clean` のため **noise は混ぜない**
- `anchor_single_channel: true` のため **Anchorのみ1ch化**（EnhPreprocessorと同じ方法で ch0 を採用）
- `speech_segment: 32000` のため **切り出しあり**（Anchor/Pos/Neg は同一切り出し）
- `contrastive_enable: true` で **Anchor / Positive / Negative を生成**
  - Positive: Anchorと **同じRIR**
  - Negative: **別RIR**（同一split内プール）
  - scaling: `contrastive_scale_min/max` (0.9–1.1) を s1/s2 それぞれ独立に適用

## 3. モデル構成

### Encoder
- STFT Encoder (`stft`)
  - `n_fft=256`, `hop_length=64`

### Spatial Encoder
- `mc_conformer_div` (`MCConformerDivSpatialEncoder`)
- `num_channels_mc=2`

### Anchorのみ1chの扱い
`espnet2/enh_se/espnet_model_npz.py` により:
- Anchorは **1ch**として `num_channels=1` で空間エンコーダへ
- Pos/Negは **2ch**として `num_channels_mc=2` で空間エンコーダへ

## 4. 損失関数（Triplet 単独）

設定ファイル: `train_se_mc_conformer_div_npz_triplet.yaml`

- **TripletLossのみ**使用
- `margin=0.2`
- L2正規化 → L2距離 → max(0, d_pos - d_neg + margin)
- Negative が複数ある場合は **平均**（`espnet_model_npz.py` で平均化）

## 5. 学習で最終的に最小化されるもの

- Anchor embedding と Positive embedding の距離を近づける
- Anchor embedding と Negative embedding の距離を **margin 以上に離す**

これにより、**空間情報（方向・響き）の差**が embedding の幾何構造に反映されることを狙う。

## 6. 何が学習されるか（想定）

### 学習される主対象
- **空間的な定位・響きの差**を捉える embedding
- 同一音声でも RIR が違うと距離が大きくなるような表現

### 期待される表現
- 同一RIR = 近い
- 別RIR = 遠い（margin以上）
- 方向差、残響差が大きいほど embedding 間距離も増大

## 7. 実行上の注意点

- `run_npz.sh` は `--stage 6` で学習のみ実行
- `batch_type: folded` のため `exp/enh_stats_8k/*/speech_anchor_shape` が必要
- Anchor を 1ch に変更したため、**shape が古い場合は Stage5 再実行推奨**

