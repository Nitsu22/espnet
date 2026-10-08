# TF-Locoformer-Split-S on WHAMR / NF-WHAMR

This recipe is derived from `../enh7_baseline`. It keeps the original S block
configuration and training settings, adding two gated Split/residual Fusion
stages. Existing datasets and statistics are reused through links; experiment
directories, checkpoints, historical run scripts, and unused data-preparation
files are not copied. New outputs are written under this recipe's `exp/`.

## Architecture

```text
STFT -> Conv + gLN
     -> Block 1 -> Split 1 -> Block 2 -> Fusion 1
     -> Block 3 -> Split 2 -> Block 4 -> Fusion 2
     -> original joint two-speaker DeConv head -> iSTFT
```

In channels-last notation, each Split maps `[B,T,F,D]` to `[B,2,T,F,D]`, then
folds the branch axis into the batch to obtain `[2B,T,F,D]`. The actual ESPnet
implementation uses `[B,D,T,F]`. Branches belonging to one example stay adjacent
in the folded batch. Blocks 2 and 4 each process both branches with shared
weights. All four blocks, and the two Split/Fusion pairs, have independent
parameters; there is no parameter sharing between stages.

The Split is a pointwise SwiGLU module: a 1x1 convolution maps `D -> 4D`,
which is divided into value and gate tensors of width `2D`. Their gated product
`value * SiLU(gate)` is projected `2D -> 2D`, then reshaped into two branches of
width `D`. This is inspired by gated speaker splitting in
[TISDiSS](https://arxiv.org/abs/2509.15666) and
[SepReformer](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d7c739c8e0383d02bb0addf8f29cd19-Paper-Conference.pdf).
It is an adaptation, not a reproduction of either model's complete splitter.

Each Fusion concatenates the processed branches into `[B,2D,T,F]`, applies a
1x1 convolution `2D -> D`, and adds the features immediately preceding that
Split. The final Fusion also returns a shared feature stream, which is decoded
by the original baseline output head. No cross-speaker attention, auxiliary
decoder, auxiliary loss, or adaptive attention fusion is introduced. The
branches are latent features and are not explicitly constrained to correspond
to individual speakers.

The S blocks retain embedding dimension 96, four heads, the original attention
dimension 128, RMSGroupNorm, frequency-then-time processing, Macaron SwiGLU FFNs
of width 128, and convolution kernel 8. The new separator has 5,310,916
parameters, versus 5,125,252 for the baseline: an increase of 185,664 (3.62%).
The TF-block workload is approximately 1.5 times the baseline because two of
the four blocks process twice the batch. Actual runtime and GPU memory must be
measured; this is not a measured speed comparison.

## Data and training settings

`dump -> ../enh1/dump` reuses noisy WHAMR, and `dump_clean ->
../enh4_clean/dump_clean` reuses NF-WHAMR. Their existing statistics are read
from `../enh1/exp/enh_stats_8k/` and `../enh4_clean/exp/enh_stats_8k/`, respectively.
The wrapper restricts execution to Stages 6-8, preserving the shared artifacts.

Both conditions preserve the baseline's mono 8 kHz input, clean anechoic
teachers, four-second training crops, variance normalization, final SI-SNR PIT
loss, seed 0, Xavier initialization, FP32, AdamW, 4000-update warmup, plateau
scheduler, and 150-epoch limit. Physical training batch is 4, accumulation is 1,
validation batch is 1, and four loader workers are used. WHAMR is the default;
select `--condition nf_whamr` for NF-WHAMR. Configuration changes relative to
`enh7_baseline` are limited to the separator choice.

## Running

Run from this directory in an appropriate compute allocation:

```bash
# Stage 6: noisy WHAMR training; repeat the command to resume.
bash run.sh --stage 6 --stop_stage 6
# NF-WHAMR uses its own configuration, data, and output directory.
bash run.sh --condition nf_whamr --stage 6 --stop_stage 6
# Stage 7: inference using valid.loss.best.pth on the allocated GPU.
bash run.sh --stage 7 --stop_stage 7
# Stage 8: CPU scoring; explicitly select the one-GPU training experiment.
bash run.sh --stage 8 --stop_stage 8 --ngpu 0 \
    --enh_exp exp/enh_train_tflocoformer_split_s_whamr_8k_1gpu_batch4
```

The default experiment names include condition, GPU count, physical batch, and
accumulation. Use a separate `--enh_exp` when changing settings. For CPU scoring
of NF-WHAMR, select its condition and
`exp/enh_train_tflocoformer_split_s_nf_8k_1gpu_batch4`. The recipe neither copies
nor initializes from `enh7_baseline` checkpoints.

## TSUBAME preparation

Synchronize the intended commit into an available checkout using the repository's
`tsubame-run` skill before submission. Existing running jobs must keep their
checkout unchanged. The prepared training script uses the existing conda
environment and CUDA 11.8 module, one full GPU (`gpu_1=1`), priority -5, and a
24-hour ceiling. A full GPU is retained because the split blocks double their
activation batch; compatibility with a half GPU has not been measured. The
24-hour limit is a resumable allocation ceiling, not a predicted completion time.

```bash
mkdir -p qsub_logs
qsub -g tga-shinoda -N tfs_split_whamr_b4 \
    qsub/train_tflocoformer_split_s_1gpu.sh whamr
qsub -g tga-shinoda -N tfs_split_nf_b4 \
    qsub/train_tflocoformer_split_s_1gpu.sh nf_whamr
```

Before Stage 6, `local/check_split_inputs.py` checks reused data/statistics,
performs a real batch-four CUDA forward/backward/AdamW step and longest-example
validation, and records `input_check.json` in the new experiment. It rejects
resume directories with a different condition, batch, or separator architecture.
The probe weights are discarded; normal training starts from seed 0 or resumes
this experiment's checkpoint. GPU inference and CPU scoring are separate stages.
No job has been submitted as part of preparing this recipe.
