# Rec-RIR 2spk PIT Baseline

## 目的

既存の1話者 Rec-RIR を壊さず、`mix_clean_reverb` を入力する2話者RIR推定 baseline を作る。

新しいRIR lossは入れない。既存 Rec-RIR と同じ `loss_cln + loss_rvb + loss_rec` を、2話者の source permutation に対して PIT 化する。

出力は mic 0 の2本のRIRだけを対象にする。

## 方針

TF-Locoformer / TF-GridNet と同じ考え方で、decoder出力を `num_spk * real_imag_dim` に増やし、speaker軸へ reshape する。

2話者、CTF length `L=60` の場合:

```text
speech decoder: C -> 2 * num_spk      = 4
reverb decoder: C -> 2 * num_spk      = 4
CTF decoder:    C -> 2 * num_spk * L  = 240
```

つまり `dim_output_spch=4` と `dim_output_CTF=240` は必要。ただしそれだけでは足りず、出力を以下のように speaker軸へ分解してPIT lossに渡す必要がある。

```text
speech/reverb:
  [B, F, T, 2 * S] -> [B, S, 2, F, T] -> complex [B, S, F, T]

CTF:
  [B, F, 1, 2 * S * L] -> [B, S, 2, F, L] -> complex [B, S, F, L]
```

## 既存1話者実装の扱い

既存1話者 Rec-RIR の model class は変更しない。

変更しない:

```text
espnet2/rir/rec_rir/model.py
espnet2/rir/rec_rir/espnet_model.py
```

2話者用は同じ package 内に追加する。

```text
espnet2/rir/rec_rir/model_pit.py
espnet2/rir/rec_rir/espnet_model_pit.py
```

`model_pit.py` は既存 `model.py` の `SpatialNetLayer` / `SpatialNetLayerNB` / `FuseLayer` を再利用し、`BiSpatialNet` 本体の1話者挙動は触らない。

## データ

確認済みの現状:

- `dump_rir_clean/raw/*_mix_clean_reverb_min_8k/wav.scp`
  - mixture入力
  - 8 kHz, 2ch
- 既存 `speech_direct.scp`
  - `spk1.scp` と同一で source 1 のみ
- 既存 `speech_reverb.scp`
  - source 1 reverb のみ
- `rir.scp`
  - `utt rir1_path rir2_path` の3列
  - scoring用
- `rir1.scp` / `rir2.scp`
  - source別RIR

2話者学習では source別scp を明示的に使う。既存single Rec-RIR用の
`speech_mix.scp`, `speech_direct.scp`, `speech_reverb.scp` は変更しない。

```text
speech_mix:     speech_mix_pit.scp  # wav.scp のコピー。2話者 mixture
speech_direct1: spk1.scp
speech_direct2: spk2.scp
speech_reverb1: spk1_reverb.scp
speech_reverb2: spk2_reverb.scp
```

`speech_mix` と全教師音声は preprocessor で channel 0 に固定する。今回の評価対象RIRも mic 0 なので、ここでmulti-channel問題には広げない。

`rir.scp` は training loss には使わず、推論後の scoring にだけ使う。

`local/prepare_rec_rir_2spk_pit_dump.sh` は既存scpを置き換えず、以下の
2話者PIT用scpだけを追加する。

```text
speech_mix_pit.scp
speech_direct1.scp
speech_direct2.scp
speech_reverb1.scp
speech_reverb2.scp
```

既存 `rir.scp`, `rir1.scp`, `rir2.scp` はすでに揃っているため書き換えない。

## Loss

1話者 Rec-RIR と同じ loss を使う。RIR waveform loss は追加しない。

```text
loss_cln = RIMag(est_spch, direct)
loss_rvb = RIMag(est_reverb, reverb)
loss_rec = RIMag(convolve(direct, est_ctf), reverb)

loss = w_cln * loss_cln
     + w_rvb * loss_rvb
     + w_rec * loss_rec
```

2話者版では、2通りの permutation で source平均 loss を計算し、合計lossが小さい方を選ぶ。

```text
perm 0:
  pred0 -> ref0
  pred1 -> ref1

perm 1:
  pred0 -> ref1
  pred1 -> ref0
```

重要:

- `loss_cln`, `loss_rvb`, `loss_rec` で別々の permutation は選ばない。
- sourceごとの3 lossを重み付き合算してから permutation を選ぶ。
- `loss_rec` では既存Rec-RIRと同じく `direct[ref]` と `est_ctf[pred]` を convolution する。
- batch内の各サンプルごとに permutation を選ぶため、lossは一度 `[B]` に保ってから最小化する。

## Scoring

既存の音声分離評価と同じ考え方にする。

ESPnetの `enh_scoring.py` は Metricごとに別々の permutation を選ばず、BSS Evalで得た1つの permutation を使ってSTOI/PESQ/SI-SNR/SDRなどを計算している。

RIR scoringでも、1 utteranceにつき1つの permutation だけを選び、その対応で全Metricを出す。MetricごとにPITをやると、RMSEとCorrなどが別対応になり、1つの推定結果として整合しない。

このbaselineでは default のPIT選択Metricを `rmse_50ms` にする。

```text
for each utt:
  pred = [rir_pred1, rir_pred2]
  ref  = [rir_ref1, rir_ref2]

  perm0 と perm1 の source平均 rmse_50ms を比較
  良い permutation を選ぶ
  その permutation で rmse / rmse_50ms / corr / RT60 / DRR / C50 を集計
```

出力:

```text
per_source.csv  # PIT後の各 source pair
per_utt.csv     # 2 source平均
summary.json    # 全 source pair平均
```

## 追加ファイル

```text
espnet2/rir/rec_rir/model_pit.py
espnet2/rir/rec_rir/espnet_model_pit.py
espnet2/train/preprocessor_rec_rir_pit.py
espnet2/bin/rec_rir_pit_inference.py
egs2/whamr/rir/conf/tuning/train_rec_rir_2spk_pit_clean_8192.yaml
egs2/whamr/rir/local/prepare_rec_rir_2spk_pit_dump.sh
egs2/whamr/rir/local/score_rir_pit.py
egs2/whamr/rir/run_rec_rir_2spk_pit_clean_8192.sh
egs2/whamr/rir/run_infer_rec_rir_2spk_pit_clean_8192.sh
egs2/whamr/rir/run_score_rec_rir_2spk_pit_clean_8192.sh
```

## 変更ファイル

```text
espnet2/tasks/rir.py
egs2/whamr/rir/rir.sh
```

変更内容:

- `rir_model_type=rec_rir_pit` を追加
- `preprocessor=rec_rir_pit` を追加
- `speech_direct1/2` と `speech_reverb1/2` を train/valid data に渡す
- `rec_rir_pit` 専用分岐を追加
- `rec_rir_pit` の入力scpは既存 `speech_mix.scp` ではなく
  `speech_mix_pit.scp` を優先する

既存の `direct` と `rec_rir` の model class とconfigは変更しない。

## 実行

学習:

```bash
cd /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir
./run_rec_rir_2spk_pit_clean_8192.sh
```

推論:

```bash
./run_infer_rec_rir_2spk_pit_clean_8192.sh
```

評価:

```bash
./run_score_rec_rir_2spk_pit_clean_8192.sh
```

## 懸念点

1. 2話者版は Rec-RIR に加えて暗黙のsource separationも必要になる。1話者Rec-RIRより難しい。
2. configは比較のためsingle版と同じ `batch_size=4` にしている。2話者PITは重いので、OOMする場合は実行時にbatch sizeを下げる必要がある。
3. 入力も出力も channel 0 / mic 0 に固定している。2ch入力を使う設計はmulti-channel Rec-RIRになり、今回のbaselineから外れる。
4. 既存 single checkpoint はそのまま resume できない。decoder出力次元が変わるため、まずscratch trainingが前提。
5. scoringのPIT選択Metricは default `rmse_50ms`。これは既存分離評価のように1つの対応を固定するための代表Metricであり、Metricごとに別PITはしない。

## 判断

このbaselineは実装可能。

目的から外れるため、以下は入れない。

```text
RIR waveform loss
training loss 用の peak alignment
training loss 用の scale normalization
multi-channel Rec-RIR
既存1話者 model class の変更
既存 checkpoint の部分ロード
```
