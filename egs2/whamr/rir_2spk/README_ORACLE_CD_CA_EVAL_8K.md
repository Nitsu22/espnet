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
