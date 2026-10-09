# Frozen 8-kHz BiMamba CTF conditioning

Run the experiment from `enh_rir` with
`run_nf8k_bimamba_ctf_film_all_s.sh`. It reuses the 16-kHz version's verified
training/inference path but selects the **separately trained 8-kHz BiMamba**.
The estimator finished normally with early stopping at epoch 56; the original
best-validation-loss checkpoint is `45epoch.pth`. Its SHA256 is pinned in
`conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_film_all.yaml`.
It is neither a 16-kHz estimator applied to 8-kHz audio nor an oracle model.

The separator matches `enh7_baseline`'s NF-WHAMR 8-kHz TF-Locoformer-S:
four blocks, embedding 96, FFN 128, STFT 256/hop 64, FP32, seed 0, physical
batch 4, accumulation 1, 32000-sample crops, full-length batch-one validation,
AdamW, and unchanged warmup/plateau/150-epoch stopping settings. The fixed
BiMamba uses its native STFT 256/hop 128/square-root-Hann. It produces
`[B, 60, 2, 129, 2]` CTF conditioning from exactly the separator's observation
or crop. Cross-attention/FiLM is applied before every separation block.
Freeze, eval mode, post-Xavier loading, identity FiLM and complete-checkpoint
restore are the same as in the 16-kHz experiment.

## Exact existing Baseline inputs

The data link is
`dump_nf_baseline_8k -> ../enh7_baseline/dump_clean`.
The original Baseline's SCPs contain recipe-relative WAV paths, so Stage 5
writes small absolute-path indexes in `exp/bimamba_nf8k_baseline_inputs`.
It retains the original IDs and exact input/reference WAV files, validates
all 28000 examples' headers and teacher lengths, and records source/output
SCP hashes. No waveform is copied, regenerated, normalized or resampled.
Source or prepared-index changes are rejected on reuse.

The `rir_2spk` training dump and these Baseline WAVs are not byte-identical:
sampled waveform differences were observed, including a validation mixture
with a larger discrepancy. Substituting that dump would therefore change
the input in a comparison with the existing Baseline. Only the pretrained
8-kHz estimator weights are taken from `rir_2spk`; conditioning is computed
from the Baseline's actual WAVs.

```bash
# CPU-only indexing/checks. Repeat safely to verify existing indexes.
bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 5 --stop_stage 5 --ngpu 0

# In a one-GPU allocation: actual pretrained-Mamba scratch update and the
# longest full-length validation example. Does not save trained weights.
python local/check_bimamba_conditioning_training.py \
  --config conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_film_all.yaml \
  --dump exp/bimamba_nf8k_baseline_inputs --input-style baseline \
  --output exp/bimamba_ctf_nf8k_cuda_check.json

# Scratch training or checkpoint resume of the same conditioned experiment.
bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 6 --stop_stage 6

# Optional matched baseline from enh_rir, with separate output.
bash run_nf8k_bimamba_ctf_film_all_s.sh --conditioning false \
  --stage 6 --stop_stage 6

# Valid/test inference and separate CPU scoring.
bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 7 --stop_stage 7
bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 8 --stop_stage 8 --ngpu 0
```

Conditioned output: `exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0`.
Optional Baseline output: `exp/enh_train_tflocoformer_s_nf8k_baseline_seed0`.
The existing `enh7_baseline` experiments are not overwritten. Both use the
same `valid.loss.best.pth` selection policy. CPU tests verify both sample rates,
strict real-checkpoint loading, native CTF shapes, same-input conditioning,
fixed weights, inference and checkpoint reload; mixing 8/16-kHz checkpoint
dimensions is rejected. Actual CUDA kernels and long training need a GPU
allocation. Implementing this experiment does not submit a training job.
