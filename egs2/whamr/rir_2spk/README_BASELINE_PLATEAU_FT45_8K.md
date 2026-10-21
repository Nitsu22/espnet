# BiMamba Baseline 8 kHz: low-LR fine-tuning from epoch 45

The original SweepCTF v2 baseline uses 5,000-update cosine warm restarts,
but each epoch contains 5,109 batches. Its validation minimum at epoch 45
and increase at epoch 46 coincide with the epoch-end LR moving from
approximately 8.9e-7 to 1e-3. This experiment tests fine-tuning without
restarts; it does not establish that restarts alone caused the increase.

- Initialize every model tensor from the original baseline's `45epoch.pth`;
  expected SHA-256: `728afa1b8d27d526ba87cb73ca48193ee08f91ee48023cc1f26d9f44bd8a990e`.
- Use a fresh AdamW optimizer, LR 1e-4, weight decay 0. This is model-only
  fine-tuning, not an exact optimizer/scheduler resume or scratch training.
- Keep the same 439,532-parameter model, spectral Sweep v2 PIT loss, seed,
  NF two-speaker dump, 4-second train crops, batching, and full-length
  validation. No time L1 or other auxiliary loss is added.
- Halve LR after three consecutive validation epochs without improvement;
  floor 1e-6. PyTorch `ReduceLROnPlateau.patience=2` implements three bad
  epochs because its comparison is strict (`num_bad_epochs > patience`).
- Stop after 15 epochs without improvement; ESPnet `patience=14` implements
  exactly 15. Maximum 100 additional epochs is a safety cap. Select by
  `valid.loss` and retain the five best checkpoints. New epoch 1 is the
  first fine-tuning epoch, after the source's 45 completed epochs.
- The source validation loss (rounded log value 0.681) remains the external
  comparison; fine-tuning is not guaranteed to improve it. Compare final
  results using the same WHAMR/LS evaluation protocol.

Launch on an available lab GPU from this worktree's recipe:

```bash
bash local/run_baseline_plateau_ft45_8k_lab.sh 0 \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
```

The launcher verifies source/model configuration compatibility and the
checkpoint hash, performs strict checkpoint loading and real GPU/PIT/RIR
checks, verifies input-shape manifests, then starts Stage 6 in a separate
output directory. GPU profiling updates are discarded before real training.
Source results are read only. It saves configuration, hashes, training log,
completion metadata, and exit state under the usual recipe `exp/` directory.
Use a persistent session on the lab host, and check GPU availability again
immediately before launch.

## Initial LR 0.01 replacement (2026-10-21)

At the user's request, the LR 1e-4 run was interrupted and retained. The
replacement starts again from the original epoch-45 weights with a fresh
optimizer and initial LR **0.01**. It does not load the interrupted run's
weights or optimizer. The three-bad-epoch halving rule, floor 1e-6,
15-bad-epoch early stopping, model, loss, and data are unchanged.

```bash
bash local/run_baseline_plateau_ft45_8k_lab.sh 0 \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk \
  lr1em2
```

This selects `train_pooled_bimamba_2spk_nf_8k_sweep_v2_ft45_plateau_lr1em2.yaml`
and writes a separate model directory ending in `_ft45_plateau_lr1em2`.
The launcher checks finite GPU updates at the requested LR before starting
Stage 6; it does not silently substitute another learning rate.

## Initial LR 0.001 replacement (2026-10-21)

After the LR 0.01 experiment stopped with invalid gradients, the user
requested initial LR **0.001**. This variant again initializes from the
original epoch-45 baseline, with a fresh optimizer. All loss, model, data,
decay, and early-stopping settings are otherwise unchanged.

```bash
bash local/run_baseline_plateau_ft45_8k_lab.sh 0 \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk \
  lr1em3
```

Outputs end in `_ft45_plateau_lr1em3`. Finalization now rejects ESPnet's
"all gradients invalid" termination even when the training process returns
zero, so that this numerical failure cannot produce a success marker.
