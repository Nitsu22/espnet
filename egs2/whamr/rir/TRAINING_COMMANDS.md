# RIR Training Commands

WHAMR RIR 推定 recipe の学習用コマンドメモです。

## 通常版

```bash
cd /net/midgar/work/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir
./run_rir.sh
```

使用 config:

```text
conf/tuning/train_rir_tflocoformer.yaml
```

## small/FLA 版

```bash
cd /net/midgar/work/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir
./run_rir_small.sh
```

使用 config:

```text
conf/tuning/train_rir_flatflocoformer_small.yaml
```

## Stage 指定

data/dump 作成のみ:

```bash
./run_rir.sh --stage 1 --stop_stage 4
```

stats 収集のみ:

```bash
./run_rir.sh --stage 5 --stop_stage 5 --skip_data_prep true
```

学習のみ:

```bash
./run_rir.sh --stage 6 --stop_stage 6 --skip_data_prep true
```

dry run:

```bash
./run_rir.sh --dry_run true
```

## データ参照元

`run_rir.sh` / `run_rir_small.sh` は次の参照元を使います。

```text
入力音声 wav:        ../se2_data/data
RIR/room param npz: ../se_npz/data
```

`../se_npz/data` の `spk1_reverb.scp` は `.npz` を指すため、`rir/data` を `se_npz/data` に丸ごと symlink しないでください。

## 主な出力先

```text
data dir:       data/tr_rir_*, data/cv_rir_*, data/tt_rir_*
dump:           dump/raw
stats:          exp/rir_stats_8k
通常モデル:     exp/rir_train_rir_tflocoformer
small/FLAモデル: exp/rir_train_rir_flatflocoformer_small
```
