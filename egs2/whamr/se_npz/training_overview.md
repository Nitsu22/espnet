# 学習内容の整理（contrastive / NPZ）

このドキュメントは、`se_npz` の contrastive 学習（Anchor/Positive/Negative）における
**現在の実装の挙動**を整理したものです。

## 1. 入力データの生成（NpzPreprocessor）

### 1.1 生成されるキー
- `speech_anchor`
- `speech_pos`
- `speech_neg1..K`（K は `contrastive_num_neg`）

### 1.2 生成の共通条件
- **s1/s2 は同一 utt の音声**を使用
- **RIR は split 内の pool から選択**（同一 split 限定）
- **スケールは s1/s2 それぞれに乱数係数を掛ける**
  - 係数は `contrastive_scale_min`〜`contrastive_scale_max` の **一様分布**
  - 設定値: 0.9〜1.1
- `wham_speech_scale` / `wham_noise_scale` は **NPZに保存された値をそのまま使用**
- `mix_type` は config で指定（現在は `clean`）
  - `clean` の場合は **noise は混ぜない**

### 1.3 Anchor / Positive / Negative の違い
- **Anchor**
  - 元の RIR（その utt の RIR）
  - 元のスケール（乱数係数=1.0）
- **Positive**
  - **同じ RIR**（Anchor と一致）
  - **スケールのみ乱数**
- **Negative**
  - **別 utt の RIR**（Anchor と異なる）
  - **Negative 同士も異なる RIR**
  - **スケールのみ乱数**

### 1.4 後処理（EnhPreprocessor 相当）
以下を **Anchor/Positive/Negative 全てに適用**:
- `speech_segment` による切り出し（現在 `32000`）
- `avoid_allzero_segment` などの既存処理
- 量子化: `PCM_16`（dump と一致させるため）

## 2. モデル構成

- Encoder: `STFTEncoder`
- Spatial Encoder: `MCConformerDivSpatialEncoder`
- 入力は **ステレオ（multi-channel）** のまま処理
- 各入力（Anchor/Positive/Negative）をそれぞれ埋め込みに変換

## 3. 損失関数とメトリクス

### 3.1 損失
- `PairwiseNegativeLoss`（temperature=0.07）
- Anchor/Positive/Negative の埋め込みを使い、**コサイン類似度**で学習

### 3.2 複数 Negative の扱い
- **各 Negative に対して loss を計算し平均**する

### 3.3 ログされるメトリクス
- `pairwise_negative_loss`
- `pairwise_negative_loss_accuracy`
  - `s_pos > s_neg` の割合（正例が負例より近い割合）

## 4. 何が学習されるか

この学習は以下の性質を **明示的に目的**としています:
- **同じ RIR** かつ **スケールのみ変化**した入力は **近い埋め込み**になる
- **異なる RIR** を与えた入力は **遠い埋め込み**になる

つまり、**スケール変動に不変で、RIR差に敏感な埋め込み**を学習する構成です。

## 5. 乱数と再現性

- 乱数は `contrastive_seed` と `contrastive_epoch` と `uid` で決定的に生成
  - 同一 epoch では再現性あり
  - epoch を変えると Anchor/Positive/Negative の組み合わせが変わる
