# Original Rec-RIR training on noisy WHAMR, native 8 kHz

Train the original two-speaker `rec_rir_pit` estimator from scratch on
`dump_noisy_2spk_8k_min`, using the existing `mix_both_reverb` observations.
This is the full original network and three speech losses (clean/direct,
reverberant, and direct-speech/CTF reconstruction; weights 1/1/1). It is not
the lightweight BiMamba, pooled Rec-RIR, or Sweep-v2-only loss variant.

The configuration preserves the original 8-kHz NF Rec-RIR architecture, STFT
256/hop128/sqrt-Hann, 60 complex CTF taps, hidden96, 6 speech/2 noise/6 CTF
layers, AdamW settings, cosine-warm-restart scheduler, AMP, maximum100 epochs
and patience10. Seed0, four-second mono crops, physical batch4/accumulation1,
validation batch1 are explicit. Input-only fold length200000 prevents full
utterance shapes from reducing physical batch4. This batching definition and
the current RIR recipe's WAVs are not claimed identical to the legacy NF run;
the two existing scores alone do not constitute a strict noise-only ablation.

Audio and RIR waveforms are never regenerated. Before CUDA starts, the runner
checks all saved index hashes, 20000/5000/3000 IDs, all mixture/teacher/noise
headers and lengths, and equality of all teacher/RIR indexes to the matching
NF dump. Sampled mixtures must equal both reverberant sources plus nonzero
noise. The native-8-kHz LS test reference must share the same IDs and physical
RIR source pairing. Only new input-length shapes and verification records are
written to this experiment's output directories.

The GPU gate uses four real training examples and the same AMP/optimizer
configuration, allowing normal GradScaler calibration. It requires two finite
optimizer updates and changed weights, native `[1,2,129,60]` CTFs, finite
two-RIR ordinary-PIM inference, and full-length validation of the longest
validation example. It records peak memory. Probe weights are discarded;
the trainer starts independently from seed0. No precision or batch reduction
is made silently when the gate fails.

Commit/push, then synchronize the dedicated checkout on the chosen lab host.
Check GPU ownership and availability immediately before starting a detached
session. Run the script from the synchronized checkout:

```bash
bash egs2/whamr/rir_2spk/local/run_rec_rir_noisy8k_lab.sh \
  0 /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
```

Results remain in the normal shared recipe:

- Model: `exp/rir_train_rec_rir_2spk_noisy_8k_all_loss`.
- Logs and validation: `exp/rec_rir_noisy8k_training_20261031`.
- Shapes: `exp/rir_stats_rec_rir_2spk_noisy_8k_all_loss`.

The detached launcher stores stdout/stderr, selected host GPU index/UUID,
source commit, start/end timestamps, phase and exit status. It redirects all
caches/temp files to its own run directory and removes them on exit. Existing
model/output directories are rejected; continuation requires an explicit,
reviewed resume rather than accidental scratch reuse.

After a normal reporter-confirmed training stop, the best single
`valid.loss.best.pth` is evaluated on all3000 noisy test mixtures. Ordinary
Sweep PIM saves two native-8-kHz, 2-second mono FLOAT RIRs per mixture. The
existing LS direct-to-reverberant reference and peak alignment/normalization/
RMSE PIT scorer are reused. Acoustic parameters of this relative-transfer
reference are diagnostics, not physical clean-to-room measurements. Results
are under the model's `evaluation_noisy_test_best/ls_score/comparison.json`;
normal completion requires `training_complete.json`, `evaluation_complete.json`
and saved exit status0, rather than the mere disappearance of a process.
