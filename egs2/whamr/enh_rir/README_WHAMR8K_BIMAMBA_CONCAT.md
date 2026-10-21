# Noisy WHAMR 8 kHz: frozen BiMamba CTF temporal conditioning, Small

New downstream separation training using DAI-92's noisy 8 kHz estimator (single
best checkpoint, epoch 45; SHA256 in the YAML). The estimator stays frozen and
receives only the actual noisy mixture/crop, never teacher speech or RIR.
Conditioning is the existing DAI-76/DAI-43 temporal-concat architecture, not FiLM:
60 frames per speaker, shared real/imag projection, 120 prefix frames before the
4 TF-Locoformer blocks, then crop the prefix before speech decoding.

Comparison: the completed `enh7_baseline/exp/enh_train_tflocoformer_s_whamr_8k_1gpu_batch4`.
Reuse its exact noisy `mix_both_reverb` observations and clean anechoic teachers,
tr/cv/tt 20,000/5,000/3,000, mono left channel, 4-second train crop, seed 0,
physical batch 4, accumulation 1, valid batch 1, four workers, FP32, AdamW and
warmup/plateau scheduling, max 150 epochs/patience 10, SI-SNR PIT. No warm start
from the noise-free separator. Initialization RNG preservation stays enabled.

Runtime setup in the dedicated TSUBAME checkout:

- `dump_whamr_baseline_8k` links to the original `enh7_baseline/dump`.
- The CPU preparation job rewrites only SCP paths into absolute form under
  `exp/bimamba_whamr8k_baseline_inputs`. It copies no WAV files and verifies all
  headers, counts, IDs and aligned lengths.
- The saved Baseline config and all six shape series must match, or preparation
  fails. Its report is `exp/bimamba_whamr8k_baseline_match.json`.
- Place the checked checkpoint at `exp/pretrained_bimamba_noisy8k/45epoch.pth`.
  Freeze/identity and SHA checks run again in the actual allocated GPU job.
- CPU and GPU jobs retain commits, exit status, allocation/memory diagnostics,
  and an actual batch-four update plus longest-validation smoke report.

From `enh_rir`, submit `qsub -g tga-shinoda qsub/prepare_whamr8k_bimamba_concat_s.sh`,
then `qsub -g tga-shinoda -hold_jid PREP_ID qsub/train_whamr8k_bimamba_ctf_concat_s_nodeq.sh`.
The first GPU job is one hour; time-limited runs preserve completed epochs.
After checking its success/checkpoint, continue with the same script using
`qsub -g tga-shinoda -l h_rt=24:00:00 qsub/train_whamr8k_bimamba_ctf_concat_s_nodeq.sh 86400 true`.
Do not change batch or precision after an OOM. Do not update this checkout while
jobs reference it. No automatic inference/test scoring is included in Stage 6.
