# Oracle-CD evaluation against raw CA RIRs

CD is clean-to-direct, DA is direct-to-all/reverberant, and CA is
clean-to-all/reverberant. This is a new evaluation protocol, separate from
ordinary clean-Sweep PIM and LS-reference scores. Only the mixture is given
to the network; the exact saved CD filter is supplied to reconstruction.

For every predicted CTF slot p and reference speaker r, convolve the original
Sweep with raw CD_r, apply the causal CTF in the STFT domain, iSTFT, and
convolve with the ORIGINAL Sweep inverse. Retain all convolution frames.
Use known-system gain calibration and fixed inverse-delay compensation.
No predicted-peak crop, normalization, or direct-filter inverse is used in
reconstruction. The reference is raw CA_r from rir_ref*.scp, with NO Sweep,
inverse Sweep, LS fitting or independently transformed CD/CA preprocessing.

Both model families' estimate_ctf() methods return reversed taps for legacy
PIM; CAReconstructor flips once to causal convolution order. Its synthesis
matches full-tail SweepCTF v2 training, with common n_fft waveform guards.
An unavoidable Sweep bandwidth limit and finite STFT/CTF approximation are
part of the end-to-end error against raw CA.

Use the same pinned legacy score_pair implementation (4e8a0f5b9b0bc5d73e3e5ace5b42d4b29b300f72),
8 kHz, 16000-sample window, peak alignment, per-RIR peak normalization,
no polarity correction, and minimum full-RIR RMSE for PIT. There are four
CA candidates [CTF slot, CD reference speaker]; CD_r and CA_r stay paired.
Choose ONE utterance-level slot permutation for all metrics. Saved WAVs
are FLOAT, in reference-speaker order, with metadata documenting oracle PIT.
Peak-aligned metrics assess shape, not absolute propagation delay or gain.
Unaligned/unscaled supplementary scores reuse the same selected permutation.
RMSE, reference-peak-onset 50 ms RMSE, correlation, RT60, DRR and C50 are
reported with finite counts. Retain all cases (including legacy RT60 outliers).
No comparison CSV is written; details are JSONL and summaries JSON.

All use full canonical FLOAT NF-WHAMR 8 kHz test mixtures, channel0,
3000 mixtures / 6000 sources, and the same physical raw CD/CA indexes.
Source index hashes, selected model/config hashes, code/scorer commits,
parameter count, actual host/GPU and completion records are preserved.

Models:
- BiMamba Baseline scratch/plateau: 165epoch.pth, 439532 parameters.
- Rec-RIR Baseline with three speech losses: 97epoch.pth.
- BiLSTM pre8/post4 plus time-Sweep L1 and plateau: 141epoch.pth,
  1090400 parameters.

Launch from the recipe:

```bash
bash local/run_oracle_cd_ca_8k_eval_lab.sh 0 \
 /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk \
 bimamba
```

Variants: bimamba, rec_rir, bilstm. Each launcher verifies a free GPU,
checks the pinned checkpoint/data hashes, runs a 2-mixture smoke test,
then the full test, and records exit status. Outputs use distinct
exp/oracle_cd_ca_VARIANT_eval8k_20261027 directories. No training changes.

## Additional DAMSEP comparison

DAMSEP uses the existing three-loss best 29epoch checkpoint transferred
from the user's TSUBAME experiment, checkpoint SHA256
f5c39e3172533fb390e026940fde0965c08c3af02d4f33b87f1a275b9d212667,
config SHA256 f923514eb16e85819070a84ca1b7a610bf153bdfdf3c53375b46de3261f61b75.
Its unmodified ESPnet adapter/vendor code is imported from commit
47d90bb4c0a7473577d0e35ca0b1c8a979b487fe (licenses retained).

Recompute CTFs from exactly the SAME canonical FLOAT NF mixture as the
other three models. Previous DAMSEP CTF exports used formatted PCM16
mixtures with quantization/clipping, so those exports are not reused.
Preserve DAMSEP's learned CTF reconstruction basis FFT512/window256/hop128/
sqrt-Hann (257 bins, 60 taps), distinct from its own network input STFT.
Convert its native causal tap order to the common helper's reversed-PIM
input order exactly once. The helper converts back to causal convolution.
All raw CD/CA indexes, original Sweep/inverse, 2s output, reference-paired
PIT and metrics are unchanged. Model training and STFT bases differ, so
this is a trained-method comparison, not an architecture-only ablation.

Use the same launcher with variant `damsep`. Results use
exp/oracle_cd_ca_damsep_eval8k_20261027. The original three-model
comparison JSON is preserved; a four-model comparison is written separately.
