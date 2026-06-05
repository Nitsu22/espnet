# Rec-RIR ESPnet implementation memo

目的: single-source Rec-RIR を WHAMR RIR recipe 上で学習・推論・評価できるようにし、TF-Locoformer baseline と同条件で比較する。

比較条件:
- input: `single_clean_reverb`
- sample rate: 8 kHz
- RIR target length: 8192 samples
- target RIR for scoring: `rir_reverberant`, source 0, mic 0
- evaluation: `local/score_rir.py` を共通使用

## 実装済み

Core:
- `espnet2/rir/rec_rir/__init__.py`
- `espnet2/rir/rec_rir/model.py`
- `espnet2/rir/rec_rir/feature.py`
- `espnet2/rir/rec_rir/pim.py`
- `espnet2/rir/rec_rir/espnet_model.py`

ESPnet integration:
- `espnet2/tasks/rir.py`
  - `rir_model_type: rec_rir` を追加
  - `preprocessor: rec_rir` を追加
  - Rec-RIR 選択時だけ `ESPnetRecRIRModel` を lazy import
- `espnet2/train/preprocessor_rec_rir.py`
  - `speech_mix`, `speech_direct`, `speech_reverb` を同一区間で crop

Recipe:
- `egs2/whamr/rir/local/prepare_rir_data.sh`
  - `single_clean_reverb` で `speech_direct.scp` と `speech_reverb.scp` を生成
- `egs2/whamr/rir/rir.sh`
  - Rec-RIR 時は `speech_mix`, `speech_direct`, `speech_reverb` で学習
  - stats dir は `exp/rir_stats_8k_rec_rir`
- `egs2/whamr/rir/conf/tuning/train_rec_rir_single_clean_8192.yaml`
- `egs2/whamr/rir/run_rec_rir_single_clean_8192.sh`
- `espnet2/bin/rec_rir_inference.py`
- `egs2/whamr/rir/run_infer_rec_rir_single_clean_8192.sh`
- `egs2/whamr/rir/run_rec_rir_single_clean_8192_all.sh`

Evaluation:
- `egs2/whamr/rir/local/score_rir.py`
- `egs2/whamr/rir/run_score_rir_single_clean_8192.sh`

## データ対応

Rec-RIR 学習入力:
- `speech_mix`: `spk1_reverb.scp`
- `speech_direct`: `spk1.scp`
- `speech_reverb`: `spk1_reverb.scp`

確認済み:
- `run_rec_rir_single_clean_8192.sh --stage 1 --stop_stage 1 --skip_train true`
- `data/tr_rir_single_clean_reverb_min_8k/speech_direct.scp`
- `data/tr_rir_single_clean_reverb_min_8k/speech_reverb.scp`

## 確認済み

静的チェック:
- `python -m py_compile`
- `bash -n`

ESPnet config:
- `rir_train --config conf/tuning/train_rec_rir_single_clean_8192.yaml --print_config`

小 subset の stats collection:
- `speech_mix_shape`
- `speech_direct_shape`
- `speech_reverb_shape`

## 実行コマンド

学習、推論、評価をまとめて実行:

```bash
cd /net/midgar/work/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir
./run_rec_rir_single_clean_8192_all.sh
```

stage 指定:

```bash
./run_rec_rir_single_clean_8192_all.sh --stage 1 --stop_stage 6  # train
./run_rec_rir_single_clean_8192_all.sh --stage 7 --stop_stage 7  # inference
./run_rec_rir_single_clean_8192_all.sh --stage 8 --stop_stage 8  # score
```

## 未解決

依存関係は保留中。

このログインノードで確認済みの問題:
- `tf-locoformer` env: `mamba_ssm` / `causal_conv1d` が未導入
- `recrir` env: `mamba_ssm` は CUDA driver 問題、`causal_conv1d` は `GLIBC_2.32` 不足

そのため、現時点で確認できた範囲は ESPnet config/preprocess/stats まで。Rec-RIR model forward、学習、PIM 推論は Rec-RIR 依存が動作する GPU 実行環境で確認する必要がある。
