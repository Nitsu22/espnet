# 8 kHz BiMamba: Sweep v2 + time-domain Sweep L1

This experiment keeps the noise-free, two-speaker 8 kHz baseline architecture
(four pre-pooling BiMamba blocks and two post-pooling Attention blocks), data,
optimizer, and scratch initialization. It adds a waveform loss to the existing
Sweep v2 RIMag objective.

For original sweep `q`, direct RIR `d_j`, reverberant RIR `h_j`, and predicted
CTF `G_i`, each candidate speaker pair uses

```text
X_j     = STFT(q * d_j)
Y_j     = STFT(q * h_j)
Yhat_ij = CTF_convolution(X_j, G_i)
yhat_ij = iSTFT(Yhat_ij)
y_j     = iSTFT(Y_j)

L_time(i,j) = sum(abs(yhat_ij - y_j)) / max(sum(abs(y_j)), epsilon)
L_pair(i,j) = L_RIMag(i,j) + lambda * L_time(i,j)
L           = mean_batch min_permutation mean_speakers L_pair
```

Both spectra are zero-padded to the same complete response extent before
synthesis; the CTF tail remains part of the loss. No inverse sweep, peak
alignment, fitted gain, or separate prediction normalization is applied.
The direct and reverberant teachers move together under one PIT assignment
chosen from the combined loss. `lambda` and `epsilon` are recorded in the
configuration as `time_sweep_loss_weight` and `time_sweep_loss_eps`.

The entry point accepts the same options as the baseline:

```bash
bash run_pooled_bimamba_2spk_nf_8k_sweep_v2_time_l1.sh \
  --stage 5 --stop_stage 6 --resume false
```

For the prepared lab data, use the guarded launcher from this recipe on an
available GPU, inside a persistent session. `RESULT_RECIPE` is the existing
recipe directory containing `dump_nf_2spk_8k_min`; it must be owned by the
current user and resolve inside an authorized nitsu directory.

```bash
RIR_LAB_PYTHON=/path/to/existing/environment/bin/python \
  bash local/run_time_sweep_l1_8k_lab.sh GPU_INDEX RESULT_RECIPE
```

The launcher refuses existing output directories, checks GPU availability
immediately before verification and training, verifies a real gradient update,
speaker-permutation invariance and two-RIR inference, prepares Stage 5
input-only batching shapes, and runs Stage 6 from scratch. It installs no
dependencies and runs no evaluation. Temporary files and caches stay under
the new run directory and are removed on exit.

Results under `RESULT_RECIPE/exp/`:

- `time_sweep_l1_training_20261019/`: GPU checks, exit status, final phase, and
  successful-completion metadata including the best checkpoint hash.
- `rir_stats_train_pooled_bimamba_2spk_nf_8k_sweep_v2_time_l1/`: batching shapes.
- `rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_time_l1/`: training logs,
  saved configuration, and checkpoints.

Compare `valid.loss_sweep_min` with the baseline Sweep loss: it independently
minimizes the spectral cost over speaker permutations, matching the baseline
metric. Total `valid.loss` contains an additional term and is not directly
comparable. The component metrics `loss_sweep` and the unweighted
`loss_time_sweep_l1` both use the selected combined PIT assignment.
Early stopping and best checkpoint selection use `valid.loss_sweep_min`,
producing `valid.loss_sweep_min.best.pth`;
`training_complete.json` records the saved configuration's criterion.
Successful completion requires
`exit_status = 0`, `final_phase = complete`, and the completion metadata.

## WHAMR and LS test evaluation

After successful training, run `local/run_time_sweep_l1_8k_eval_lab.sh
GPU_INDEX RESULT_RECIPE` from a persistent session on an idle lab GPU.
It verifies the saved best-checkpoint hash, full test indexes and the
Sweep/inverse hashes against the immutable 8 kHz LS preparation. Inference
uses the native FLOAT mixture, full utterance, channel 0, ordinary Sweep PIM,
and FLOAT two-second RIRs. The same 3,000 predicted pairs are scored against
physical WHAMR RIRs and the LS relative-transfer reference, using RMSE PIT,
peak alignment, peak normalization and no polarity alignment. Both reports
include the 50 ms interval starting at the reference peak.

Results are saved under `RESULT_RECIPE/exp/time_sweep_l1_eval8k_20261019/`:
`whamr/summary.json`, `ls/models/00/summary.json`, prediction metadata and
`evaluation_complete.json`. The existing scoring implementation is pinned
to commit `4e8a0f5b9b0bc5d73e3e5ace5b42d4b29b300f72` in
`RESULT_RECIPE/exp/frequency_ls_eval_code_20261014`. No comparison CSV is
written. Historical runs which re-encoded input mixtures to PCM16 have a
different input condition; their numbers are not a strictly matched control.
