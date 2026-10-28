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
