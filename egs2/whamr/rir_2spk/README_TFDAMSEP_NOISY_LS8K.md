# TF-DAMSEP versus noisy BiMamba: common LS evaluation

Compare the completed TF-Locoformer + DAMSEP three-loss WHAMR experiment
(best epoch 138) with native 8 kHz, two-speaker noisy BiMamba (best epoch 45).
Both use the exact formatted PCM16 mixture that TF-DAMSEP saw on TSUBAME,
full utterances, channel 0. No training or checkpoint selection is performed.

The existing immutable native 8 kHz direct-to-reverb LS reference is reused:
causal FIR 16000 samples, 2 seconds, ridge 1e-6, sweep 8.192 seconds.
The noisy input's four physical-RIR SCP hashes must match the LS provenance.
Every formatted input and saved CTF is checked against SHA256 recorded at
the TSUBAME source. Quantization/saturation relative to the FLOAT noisy dump
is audited, and the two speakers plus additive noise are checked independently.

Ordinary-sweep PIM converts each model using its own training transform:
TF-DAMSEP FFT512/window256/hop128/sqrt-Hann with causal tap-zero ordering;
BiMamba uses its existing inference API and FFT256/window256/hop128/sqrt-Hann.
No frequency interpolation or oracle direct-RIR excitation is used. The tested
`damsep_pim.py` implementation is reused from `codex/damsep-ls8k-eval`.

Both output mono FLOAT RIRs at 8 kHz and 16000 samples. The common scorer uses
peak alignment, independent peak normalization, full-RMSE two-speaker PIT,
and retains polarity. Primary metrics are full-waveform RMSE and correlation.
Secondary diagnostics are RMSE in the 50 ms after the reference peak and late
NMSE (50 ms after that peak through the end of the two-second window).
The same full-RMSE PIT assignment is retained for every secondary metric.
Differences are BiMamba minus TF-DAMSEP, with a paired percentile bootstrap
(10000 draws, fixed seed 20261028). Both sources stay within a mixture unit.
Intervals characterize this test set conditional on the fitted checkpoints,
not training-seed variability or a causal architecture effect.

The LS reference measures a relative transfer. Absolute delay/gain are not
scored. RT60/DRR/C50 from the common scorer are relative-filter diagnostics,
not physical room parameters. Training losses and model sizes differ.

Run on an available lab GPU from a committed, pushed checkout. The launcher
records host, commit, dates, exit status and cleans its own temporary directory.
It rejects an existing output directory. Run the explicit three-mixture smoke
test successfully before the full evaluation:

```bash
bash egs2/whamr/rir_2spk/local/run_tfdamsep_noisy_ls8k.sh \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/tfdamsep_noisy_bimamba_ls8k_20261028/input \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/tfdamsep_noisy_bimamba_ls8k_20261028/smoke 3
```

For full evaluation use a new output directory and limit 0. Inspect the saved
exit status, `complete.json`, input audit, model scoring counts (3000/6000),
and `paired_comparison.json` before reporting completion.

## Completed evaluation, 2026-10-28

The three-mixture smoke test and full 3000-mixture/6000-source evaluation both
exited with code 0 on tooru, RTX 4090 GPU0. All 12000 output headers and all
primary/late metrics were independently validated. Six numerical/unit tests
passed. Evaluation code: `a665e78daccd25a7e0b183882c8a98917de772e6`.

| Metric | TF-DAMSEP, best138 | Noisy BiMamba, best45 |
|---|---:|---:|
| Full waveform RMSE, lower is better | 0.00318570 | 0.00327544 |
| Full waveform correlation, higher is better | 0.948946 | 0.943547 |
| First 50 ms RMSE, lower is better | 0.01730779 | 0.01794674 |
| Late NMSE, dB, lower is better | -1.719534 | -2.138706 |

TF-DAMSEP reduces full RMSE by 2.74%; BiMamba better preserves the late
relative-transfer waveform. Paired 95% intervals for BiMamba minus TF-DAMSEP:
RMSE `[0.00007501, 0.00010520]`, correlation `[-0.00604345, -0.00477633]`,
early RMSE `[0.00054277, 0.00073951]`, late NMSE `[-0.46824, -0.37222]` dB.
These conditional test-set differences do not establish a training-seed or
architecture effect. Model sizes and training/model-selection objectives differ.

Results, provenance, individual scores and the final validation are stored in
`exp/tfdamsep_noisy_bimamba_ls8k_20261028/RESULTS.md`, `final_checks.json`, and
`full/paired_comparison.json`. The input audit found 142 mixtures with existing
PCM16 saturation (1415 samples over both channels); both models received the
same saturated input, matching the original TF-DAMSEP input SHA256 exactly.
