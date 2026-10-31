# Original Rec-RIR conditioning on the NF-WHAMR batch-four baseline

This run matches `enh7_baseline`'s NF-WHAMR 8-kHz TF-Locoformer-S: four
blocks, embedding 96, FFN 128, 256/64 STFT, FP32, seed 0, one GPU, physical
batch 4, accumulation 1, four-second training crops, full-length batch-one
validation, AdamW, unchanged warmup/plateau and 150-epoch/patience-10 rules.
It uses the exact existing mixture and anechoic teacher WAVs. It does not
resume the older batch-eight Rec-RIR-conditioned separator.

The original native-8-kHz two-speaker Rec-RIR PIT estimator is frozen in eval
mode. Its original best-validation-loss artifact is `97epoch.pth`, SHA256
`5fe27c7996e70d290d8646fb39f9cb6fe6a3f2972138e5f16b42730c8f4cafc8`.
The configuration embeds the source experiment's complete `model_conf`.
This is the original three-loss Rec-RIR, not pooled BiMamba or Sweep-v2.

CTF estimation sees precisely the separator's mono observation/crop, without
other utterances' padding. The original `estimate_ctf` API supplies peak
normalization, complex CTFs and causal tap order. No second tap flip, teacher
matching, oracle information or CTF amplitude normalization is added.
The resulting `[B,60,2,129,2]` condition uses the established Cross-Attention
and identity-initialized FiLM before each separator block. Predictor creation
preserves the CPU RNG; pinned weights are strictly restored after Xavier.
Complete separation checkpoints embed the frozen predictor and provenance.

The dedicated checkout has these external, read-only artifacts:

- `assets/rec_rir_2spk_8k_97epoch.pth`: a verified copy of the original model.
- `exp/bimamba_nf8k_baseline_inputs`: a link to the already verified absolute
  baseline indexes. Readiness checks original and prepared index hashes and
  all 20,000/5,000/3,000 counts. Original WAVs are shared without modification.
- `dump_nf_baseline_8k`: link to the original NF baseline dump, required by
  the shared runner. The existing baseline recipe's dump links remain valid.

Run from `enh_rir` in the dedicated checkout:

```bash
python local/check_recrir_readiness.py
qsub -g tga-shinoda -p -5 -l cpu_4=1 -l h_rt=00:10:00 \
  -N recrir_control qsub/nf8k_recrir_pipeline.sh --phase control
```

The controller first submits a `gpu_1=1` 24-hour job. Before training it
requires an exact public-CTF-API check, real batch-four FP32 forward/backward/
AdamW step, unchanged frozen weights, and the longest validation utterance.
Probe weights are discarded. A failed probe stops the chain without changing
batch size or precision. `cuda_check.json` records peak memory and provenance.

Only a scheduler-confirmed walltime termination with checkpoint epoch progress
can resume training automatically. Other failures stop the chain. The controller
requires the actual reporter stopping criterion and consistent best checkpoint
before inference. Stage 7 is a separate one-GPU job, Stage 8 an eight-worker
CPU job. Both splits use `valid.loss.best.pth`; WAV peak normalization is on,
matching the baseline and preventing PCM16 clipping. The condition is estimated
again from the actual full utterance at inference time.

Output: `exp/enh_train_tflocoformer_s_nf8k_recrir_ctf_film_all_seed0`.
History: `.pipeline/jobs.json`; completion: `.pipeline/score.done.json` and
`RESULTS.md`. A time limit is not a training-completion prediction.

At priority -5, per-allocation upper estimates are 3.84 points for GPU training
(24 h), 0.64 for GPU inference (4 h), 0.144 for CPU scoring (6 h), and 0.002 for
each CPU controller (10 min). Runtime is not yet measured. Retry limits are
12 training segments, 2 inference jobs and 3 scoring jobs; using every retry
would be at most 47.828 points including controllers. These are upper estimates,
not confirmed charges; successful stages do not trigger their retry allowance.
