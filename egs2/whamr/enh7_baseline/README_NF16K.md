# Native 16 kHz NF-WHAMR TF-Locoformer-S baseline

Use the existing `rir_2spk/dump_nf_2spk_16k_min` mixture and
`speech_direct1/2.scp` (the generator's `s1/2_anechoic` WAVs). These are
native 16 kHz recordings; no 8 kHz upsampling or audio duplication is used.
Speaker file pairs match the 8 kHz corpus, but utterance IDs/gains and room
conditions have not been shown identical. Do not interpret an 8-vs-16 kHz
score difference as a sample-rate-only ablation.

`qsub/prepare_nf16k_baseline.sh` runs on CPU. It checks all mixture, clean,
and reverberant teacher headers (28,000 utterances), verifies sampled mixture
sums, and creates separate `dump_nf16k/raw/*_mix_clean_reverb_min_16k` SCPs.
Audio remains at its existing absolute path. It writes mono sequence lengths
for all three supervised fields to `exp/enh_stats_16k/{train,valid}`.
These replace Stage 5's length statistics only: the configuration uses no
global feature normalization or feature mean/variance files. The readiness
manifest is written last and is required by the GPU job.

`qsub/train_nf16k_tflocoformer_s_1gpu.sh` checks SCP hashes and shapes, runs a
real CUDA batch-four update and full-length validation probe, then starts
Stage 6. Probe weights are discarded; training independently initializes
from seed 0. The model and optimizer follow the 8 kHz S baseline, with
STFT 512/hop 128 (32 ms/8 ms), 64,000-sample crops (4 seconds), FP32,
one GPU, batch 4, accumulation 1, validation batch 1, and fold length 400,000.
Statistics must not be reused from the 8 kHz run.

Run the jobs from this recipe in a separate TSUBAME checkout while 8 kHz
jobs are active. Submit CPU preparation first; submit training after verifying
the readiness manifest (or use `-hold_jid` plus the manifest guard).
Output: `exp/enh_train_tflocoformer_s_nf_16k_1gpu_batch4`, including
`input_check.json`, `train.log`, `checkpoint.pth`, `valid.loss.best.pth`.
The 24-hour GPU allocation may require checkpoint resume; evaluate only after
training finishes using Stage 7 and CPU Stage 8 of
`run_nf16k_tflocoformer_s.sh`.
