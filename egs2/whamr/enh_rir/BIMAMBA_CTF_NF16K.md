# Frozen BiMamba CTF conditioning on native 16-kHz NF-WHAMR

All runs and outputs in this experiment live in `enh_rir`. The data link is
`dump_nf_2spk_16k_min -> ../rir_2spk/dump_nf_2spk_16k_min`. No audio or RIR is
copied, resampled, overwritten or passed as an oracle condition. The same
native noise-free mixture, channel 0, and the same two anechoic speech teachers
are used by the Baseline and conditioned model. Train/valid/test contain
20000/5000/3000 utterances.

The separation architecture and training configuration match
`enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_16k.yaml`: TF-Locoformer-S
with four blocks, embedding 96, FFN 128, STFT 512/hop 128, FP32, seed 0,
physical batch 4, accumulation 1, 4-second training crops, batch-one full-length
validation, AdamW and the same warmup/plateau/150-epoch stopping settings.
Baseline and conditioned runs have separate output/checkpoint directories.

The standard four-pre-block/two-post-block pooled BiMamba Sweep-v2 estimator
uses its trained `44epoch.pth`, selected by the original best-validation-loss
link when this experiment was implemented. Its SHA256 is pinned in the YAML.
This is **estimated CTF**, not oracle CTF/RIR. BiMamba receives only the same
mixture/crop as the separator; it is always in eval mode with no gradients.
It uses its original STFT 512/hop 256/square-root-Hann and peak normalization.
Each utterance is processed without other utterances' padding. Two complex
CTFs of 60 taps/257 bins are converted to `[B, 60, 2, 257, 2]` in the existing
public inference tap convention, preserving speaker slots.

The separator projects real/imaginary CTF features and applies cross-attention
plus FiLM before each of its four blocks. CTF taps serve as conditioning tokens;
the 16-ms CTF tap grid is not used to convolve the separator's 8-ms STFT frames.
FiLM is identity-initialized after global Xavier initialization. Pretrained
BiMamba weights are then loaded, so global initialization cannot erase them.
Full separation checkpoints include the frozen predictor, readiness flag and
source hash; inference does not require the original BiMamba checkpoint once
a complete separation checkpoint is loaded. The Mamba package/backend remains
a runtime dependency. A scratch forward fails if no source weights were loaded.

Run from `egs2/whamr/enh_rir` inside the appropriate compute allocation:

```bash
# CPU-only data/path/ID/length checks (sampled waveform/sum checks).
bash run_nf16k_bimamba_ctf_film_all_s.sh --stage 5 --stop_stage 5 --ngpu 0

# Before long training: actual pretrained Mamba batch-four FP32 CUDA update
# and longest full-utterance validation. No trained weights are saved.
python local/check_bimamba_conditioning_training.py \
  --output exp/bimamba_ctf_conditioning_check.json

# Train from scratch, or resume the same output when a checkpoint exists.
bash run_nf16k_bimamba_ctf_film_all_s.sh --stage 6 --stop_stage 6

# Optional matched baseline from this recipe with the identical linked input.
bash run_nf16k_bimamba_ctf_film_all_s.sh --conditioning false \
  --stage 6 --stop_stage 6

# Full-utterance valid and test inference, then CPU scoring.
bash run_nf16k_bimamba_ctf_film_all_s.sh --stage 7 --stop_stage 7
bash run_nf16k_bimamba_ctf_film_all_s.sh --stage 8 --stop_stage 8 --ngpu 0
```

The default conditioned output is
`exp/enh_train_tflocoformer_s_nf16k_bimamba_ctf_film_all_seed0`; the optional
Baseline output is `exp/enh_train_tflocoformer_s_nf16k_baseline_seed0`.
Source checkpoint/hash changes require a new config and experiment directory.
Stages 7/8 use `valid.loss.best.pth`, the same selection policy as the Baseline.
Scoring reads the original anechoic references and channel 0; validation scores
and test scores must be reported separately.

CPU verification covers source preservation after initialization, freezing,
true-length/tap/slot layout, finite separation gradients, identity FiLM parity,
complete checkpoint restore, bad-source failure, native-16k config equality,
and full/segmented inference. The real pretrained state dict was strictly
loaded and compared exactly. CUDA kernels and long training must be verified
in a GPU allocation; preparing this experiment does not submit a GPU job.
