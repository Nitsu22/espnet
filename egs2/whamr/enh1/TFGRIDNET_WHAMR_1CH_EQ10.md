# WHAMR! 1ch TF-GridNet, equation (10)

Target: TASLP 2023 Table VIII, DNN1 SI-SDR-SE+MC, SI-SDR 11.0 dB
(SI-SDRi 17.1 dB given the paper's -6.1 dB observation score).
Source: https://zqwang7.github.io/publications/TASLP2023_TF-GridNet.pdf
This is a reproduction attempt, not a claim of matching the published result.

- Original ESPnet `tfgridnet` (not V2), B4/H192/D24/I8/J1, four attention
  heads, E4; 5,425,552 trainable parameters. No RIR input, DNN2 or MFWF.
- 8 kHz/min WHAMR, `mix_both_reverb`, microphone 0. Corresponding
  `s1_anechoic` / `s2_anechoic` direct-path targets retain their delay.
- STFT 256, hop 64, sqrt-Hann analysis and synthesis.
- Equation (9) scales estimates, with no mean subtraction. Equation (10)
  adds mean absolute error between summed **scaled estimates** and summed
  direct-path targets. Joint utterance PIT minimizes the whole loss. Speaker
  SI-SDR-SE costs are summed, not averaged. MC coefficient is 1.
- Input sample-variance normalization (ddof=1) excludes padded samples;
  loss uses the same mixture scale for predictions and references and excludes
  padding. An opt-in flag enables this normalization without altering defaults.
- Numerical choices: variance floor 1e-16; energy floors 1e-8 for alpha,
  reference energy and residual energy; no detached alpha or half precision.
- One random segment of up to 4 s per mixture per epoch. All 20,000 training
  utterances are retained, including short ones. Full 5,000-utterance validation;
  3,000 test utterances are excluded from training/model selection. No DM or
  augmentation. ESPnet's common peak scaling, when needed, cancels under the
  subsequent shared mixture standard-deviation normalization.
- Adam LR 1e-3, eps 1e-8, weight decay 0; gradient clip L2 norm 1;
  ReduceLROnPlateau factor 0.5, PyTorch patience 3 (the public ESPnet convention).
  PyTorch reduces after more than three bad validation checks, which is an
  implementation detail relative to the paper's wording of three epochs.
- Explicit fallback choices, not paper-established WHAMR values: seed 0,
  Xavier uniform, FP32/no AMP, deterministic cuDNN, batch 4 on one GPU,
  validation batch 1, max 150 epochs, early stopping patience 5,
  best validation-loss checkpoint only (no checkpoint averaging).

## Execution and outputs

Pre-launch CPU checks passed: independent NumPy least-squares equation oracle,
joint PIT assignment (including a case where MC changes the assignment),
padding/batch reduction, permutation and scale invariance, finite-difference
gradient check, silent/perfect signals, and length-aware mixture normalization.
The test functions were run directly with `runpy` because pytest is absent in
the existing environment. The exact 5,425,552-parameter model also passed a
real-data four-item 1,024-sample CPU update and checkpoint round-trip.

The tensor GPU 3 smoke test also passed 12 real-data 4-second/batch-4 updates,
all-parameter finite gradients, full longest-validation input (122,907 samples),
permutation/scale invariance and checkpoint restoration. Warm update median was
2.364 seconds; peak allocated/reserved memory was 25.888/26.416 GiB. This repeats
one minibatch and is not an end-to-end epoch timing or a convergence result.

Host: tensor, physical GPU index 3 (A100 40GB). Python:
`/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python`.
Code is isolated in the `codex/tfgridnet-whamr-eq10-20261101` worktree,
based on previously pushed commit `58ee8930b3`, with experiment changes on top.
The existing checkout and its unpushed commit are not modified.

Output directory:
`/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/enh1/exp_tfgridnet_whamr_1ch_eq10_20261101`.

`local/tfgridnet_whamr_eq10.py prepare` creates private absolute-path SCPs and
shape metadata from existing lengths (the same waveform is used for mixture
and reference lengths after checking keys and sampled headers). Original data
is read only. Every waveform path is checked, and first/middle/last/longest
waveform headers per split are checked. SCP hashes are saved in `data_audit.json`.

`run_tfgridnet_whamr_1ch_eq10.sh` requires a private task-specific `/tmp/nitsu/`
TMPDIR and a detached tmux session. It checks GPU availability, performs an
exact-model real-data forward/backward/optimizer/checkpoint smoke test, including
the longest validation utterance, and starts fresh training. Smoke-test weights
are never used to initialize training. Temp/cache files are removed on exit.

- `run.status`, `launcher.log`: lifecycle and exit status.
- `code_commit.txt`, `host.txt`, `gpu_index.txt`: provenance.
- `smoke/cpu.json`, `smoke/gpu.json`: model checks and measured GPU memory/timing.
- `seed0/train.log`, `seed0/config.yaml`, `seed0/checkpoint.pth`: training.

After normal training termination, the best checkpoint is evaluated on dev/test
and the observed mixture is scored too. SI_SNR (ESPnet's standard SI-SDR), SDR,
STOI and ESTOI are aggregated, plus narrow-band PESQ if its package is installed.
The current environment lacks PESQ; this metric remains unmeasured unless an
approved evaluation environment is provided later. STOI/ESTOI are percentages;
divide ESTOI by 100 for the paper's 0-1 scale. Improvement scores must subtract
the observation score measured by the same evaluation pipeline. Saved estimates
are peak-normalized to prevent clipping of the scale-invariant loss outputs.
All-invalid-gradient termination is treated as failure even if ESPnet exits 0.
