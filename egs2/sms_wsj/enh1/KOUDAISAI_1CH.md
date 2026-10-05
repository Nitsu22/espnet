# Koudaisai TF-GridNet 1ch ablation

This experiment starts from random initialization. It removes DOA and selects
microphone index 0 from Roland's four-microphone Koudaisai dump. Original files
are read only. It does not use the earlier SMS-WSJ paper-reproduction objective.

## Reference and comparison

Reference checkpoint on the lab filesystem:
`/net/fractal/work/roland/research/tools/espnet/egs2/sms_wsj/enh1/exp4/enh_train_enh_tfgridnet_tf_lr-patience3_patience5_multitask_4mics_koudaisai_raw/30epoch_copy.pth`.
Both checkpoint block indices (`0`, `1`) and `config.yaml` confirm two blocks.

Preserved settings: H=192, D=48, I=4, J=1, four attention heads, 8 kHz,
Hann window, FFT 256/hop 64, 4-second chunks with 50% overlap, seed 0,
Adam 0.001/epsilon 1e-8/no weight decay, gradient clip 5, float32/no AMP,
plateau LR factor 0.5/patience 3, early stopping patience 5, maximum 50 epochs,
and retaining 50 best epochs. Global batch 4 uses one chunk per rank with 4 GPUs.

The separation loss is 0.95 * (speaker-mean PIT + mixture-constraint loss).
Each term includes waveform, real, imaginary, and magnitude L1. Spectral losses
use raw predicted spectra before iSTFT, not the re-analyzed output waveform.
MC compares sums of references and predictions. The DOA loss/labels/head are
absent. The 0.95 separation coefficient is intentionally preserved.
The smoke check compares values AND gradients against Roland's original solver
with the DOA coefficient set to zero.

The first-channel mixture standard deviation normalizes both input and targets.
The original used the standard deviation over all four input microphones.
This normalization change is inherent to the 1ch ablation. Standard TF-GridNet
restores output scale; the loss removes that scale again, matching the original
normalized-unit objective. Waveform output normalization remains an inference
option, independent of the objective.

Unlike the original single-GPU run, DDP partitions utterances and chunk caches
across ranks. Global batch size is preserved; sample order and discarded final
chunks are not guaranteed to be identical. DOA-free validation loss also changes
model selection. For an epoch-matched comparison with `30epoch_copy.pth`, use
`30epoch.pth` if training reaches epoch 30; do not silently substitute an average.

## Data

Source dump: `/net/ox/data5/roland/research/sms_wsj_dump_4mic`.
Destination on TSUBAME:
`/gs/bs/tga-shinoda/nitsu/data/sms_wsj_4mic_koudaisai/dump_4mic`.

Only audio referenced by `wav.scp`, `spk1.scp`, `spk2.scp` for train/valid/test
and necessary metadata are copied. All four audio channels remain byte-identical;
the preprocessor selects channel 0 for both mixture and references. No relabeling
or regeneration of targets is performed. Do not infer direct-path targets from
directory names; this ablation keeps whatever references the original dump uses.

`local/transfer_koudaisai_tsubame.sh` runs on tensor in tmux, transfers real audio
files via rsync, audits contents with rsync --checksum, verifies sizes/IDs/paths,
and then writes `COPY_VERIFIED.json`. The source is never modified.
The destination is TSUBAME project storage, not a symlink to lab storage.

## Preparation and later training

From `egs2/sms_wsj/enh1` on TSUBAME after Git synchronization and data verification:

```bash
mkdir -p qsub_logs
qsub -g tga-shinoda qsub/koudaisai_1ch_prepare.sh
```

Preparation allocates `node_f=1`, checks a disposable random model with a
4-GPU forward/backward (zero optimizer steps, no model checkpoint), then runs
only Stage 5 (statistics). The default `run_tfgridnet_koudaisai_1ch.sh` also stops
at Stage 5. Inspect `exp_tfgridnet_koudaisai_1ch/prepare.status`, `smoke_ddp.log`,
`stats.log`, and `enh_stats_8k/{train,valid}/speech_mix_shape` before training.

Training is a SEPARATE, user-authorized submission of
`qsub/koudaisai_1ch_train.sh` with an appropriate `qsub -l h_rt=...` walltime.
It uses `node_f=1`, four GPUs, and Stage 6 only. It is not automatically submitted
by any preparation or transfer script. Models will be saved under
`exp_tfgridnet_koudaisai_1ch/tfgridnet_2block_1ch`.

CPU validation on the lab host:

```bash
PYTHONPATH=../../.. PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
NUMBA_CACHE_DIR=/tmp/nitsu_koudaisai_numba OMP_NUM_THREADS=1 \
/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python \
local/check_tfgridnet_koudaisai.py \
--roland-root /net/fractal/work/roland/research/tools/espnet
```
