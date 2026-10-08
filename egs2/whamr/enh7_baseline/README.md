# WHAMR / NF-WHAMR baselines

This directory holds new baseline experiments for comparisons with TF-based
DAMSEP. It was scaffolded from `../enh_tmp`: `cmd.sh`, `local/`, scheduler
configuration, and the original TEMPLATE symbolic links are reused. Historical
experiments, dumps, data directories, MEMO, MVDR results and `separate.py` were
not copied. `run.sh` links to the new `run_tflocoformer_s.sh`.

## TF-Locoformer-S

The configuration comes from the **saved config of the completed experiment**
`../enh4_clean/exp/enh_train_enh_tflocoformer_nocashe_clean_small_4gpu/`, rather
than the historical run script's stale configuration filename. It preserves
the four-block, embedding-96, `tflocoformer_nocashe` network, clean SI-SNR PIT
loss, variance normalization, seed 0, Xavier initialization, FP32, AdamW,
4000-update warmup, plateau scheduler and 150-epoch limit. The old test scores
are SI-SNR 18.28 dB / SDR 19.45 dB; these are reference results, not results
from this new recipe.

Defaults are one allocated GPU, physical training batch 4, accumulation 1,
validation batch 1, four loader workers, no attention plotting, and logging
every 50 updates. `--condition nf_whamr` is the default; `--condition whamr`
selects noisy WHAMR with the same model/optimization settings. The two config
files differ only in their descriptive comments.

## Reused artifacts and separate outputs

- `dump -> ../enh1/dump` exposes noisy WHAMR; `dump_clean ->
  ../enh4_clean/dump_clean` exposes NF-WHAMR. Both are existing stereo 8 kHz
  min WAV/SCP files with anechoic clean teachers. No noise source is passed
  as a separate training target: noisy WHAMR uses the original noisy mixture.
  The preprocessor selects the left channel and
  crops training utterances to at most four seconds; short examples remain.
- Stage 6 reads the original three-series train/valid statistics from
  `../enh4_clean/exp/enh_stats_8k/` for NF-WHAMR and
  `../enh1/exp/enh_stats_8k/` for WHAMR. No experiment/checkpoint directory is copied.
  The wrapper restricts execution to Stages 6-8 so shared statistics and dumps
  cannot be regenerated through it.
- New training, inference and scoring outputs go to this recipe's
  `exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4/` and
  `exp/enh_train_tflocoformer_s_whamr_8k_1gpu_batch4/`. Condition/GPU/batch/accumulation
  determine separate default output paths; `--enh_exp` can override them.
  Do not reuse an existing experiment for a different setting.
- Fold length is set to 200000 samples: the existing full-utterance shape
  files reach 130034 samples, although training crops to 32000. This prevents
  folded batching from shrinking physical batch 4 on one GPU. The wrapper
  verifies the training shapes stay below that threshold. The old four-GPU
  run enforced minimum global batch 4 through distributed sampling.

Run from this directory in an appropriate compute allocation:

```bash
# Stage 6: clean-only baseline training, from scratch on the first invocation.
bash run.sh --stage 6 --stop_stage 6
# Noisy WHAMR, with the same batch/optimization settings.
bash run_whamr_tflocoformer_s.sh --stage 6 --stop_stage 6
# Resume the same experiment by repeating the command.
# Alternative effective-batch-4 run; keep its checkpoint directory separate.
bash run.sh --batch_size 1 --accum_grad 4 \
    --enh_exp exp/enh_train_tflocoformer_s_nf_8k_1gpu_accum4
# Stage 7: one inference process on an allocated GPU.
bash run.sh --stage 7 --stop_stage 7
# Stage 8: scoring in a separate CPU allocation.
bash run.sh --stage 8 --stop_stage 8 --ngpu 0
```

The default inference checkpoint is `valid.loss.best.pth`, matching the older
baseline's selection policy. Do not change batch settings in an existing
experiment: use a new `--enh_exp`. `CUDA_VISIBLE_DEVICES` is left to the
allocation. TSUBAME job submission and synchronization follow the repository's
`tsubame-run` skill.

## TSUBAME production jobs

After synchronization, submit from this recipe directory:

```bash
mkdir -p qsub_logs
qsub -g tga-shinoda -N tfs_whamr_b4 qsub/train_tflocoformer_s_1gpu.sh whamr
qsub -g tga-shinoda -N tfs_nf_b4 qsub/train_tflocoformer_s_1gpu.sh nf_whamr
```

Each job requests one full H100 (`gpu_1=1`), eight CPU cores, 96 GB host RAM,
priority -5 and a 24-hour ceiling. Each performs Stage 6 only; GPU inference
and CPU scoring are separate stages. The existing tf-locoformer conda
environment and CUDA 11.8 module are used without installing dependencies.

Within the charged allocation, `local/check_baseline_inputs.py` validates all
SCP/statistics IDs, counts and shape alignment; samples the rates, channels,
waveform lengths and clean-teacher equivalence; and verifies sampled noisy
mixtures equal NF mixtures plus the existing noise images for train/valid.
Test dumps do not require isolated noise files. The check then runs a real
four-example, four-second FP32 CUDA forward/backward/AdamW update and full-length
validation of the longest example. It records data hashes and GPU memory in
the experiment's `input_check.json`. This probe does not save its model weights;
normal training starts from seed 0 or the experiment's existing checkpoint.

On a 24-hour time limit, resubmit the same condition to resume from the last
completed epoch; an interrupted epoch repeats. The 150-epoch limit and early
stopping remain the saved baseline's settings. Runtime for the new physical
batch-four setup must be measured rather than inferred from GPU count alone.
At priority -5 the walltime-based estimate is at most 3.84 points per initial
job, or 7.68 for both: `1 * 0.2 * 1 * 0.8 * 24`. These are estimates, not
confirmed charges; subsequent continuation jobs are additional allocations.
