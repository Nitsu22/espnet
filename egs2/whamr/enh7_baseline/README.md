# NF-WHAMR baselines

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
every 50 updates. Batch-4 memory/speed on one H100 is **not yet measured**.
No training job has been submitted as part of creating this recipe.

## Reused artifacts and separate outputs

- `dump_clean -> ../enh4_clean/dump_clean` exposes the existing stereo 8 kHz
  min NF-WHAMR WAV/SCP files. The preprocessor selects the left channel and
  crops training utterances to at most four seconds; short examples remain.
- Stage 6 reads the original three-series train/valid statistics from
  `../enh4_clean/exp/enh_stats_8k/`. No experiment/checkpoint directory is copied.
  The wrapper restricts execution to Stages 6-8 so shared statistics and dumps
  cannot be regenerated through it.
- New training, inference and scoring outputs go to this recipe's
  `exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4/`, specified explicitly with
  `--enh_exp`. The existing baseline's results are not overwritten.
- Fold length is set to 200000 samples: the existing full-utterance shape
  files reach 130034 samples, although training crops to 32000. This prevents
  folded batching from shrinking physical batch 4 on one GPU. The wrapper
  verifies the training shapes stay below that threshold. The old four-GPU
  run enforced minimum global batch 4 through distributed sampling.

Run from this directory in an appropriate compute allocation:

```bash
# Stage 6: clean-only baseline training, from scratch on the first invocation.
bash run.sh --stage 6 --stop_stage 6
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
`tsubame-run` skill; scheduler scripts can be added after memory/speed checks.
