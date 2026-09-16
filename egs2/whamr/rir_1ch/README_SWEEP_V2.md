# TF-Locoformer Sweep CTF v2

v2 learns a direct-to-reverberant CTF using paired physical RIRs, with no speech
teacher and no precomputed inverse/deconvolved RIR:

```text
direct_sweep = sweep * rir_direct   # clean -> direct RIR
reverb_sweep = sweep * rir_ref      # clean -> reverb RIR
prediction = STFT(direct_sweep) convolved over frames with estimated CTF
loss = mean(abs(real error) + abs(imag error) + abs(magnitude error))
reference = STFT(reverb_sweep)
```

The network input remains single-speaker reverberant speech. The RIR pair is
used only for training/validation supervision, not as an inference input.
The dump already provides `rir_direct.scp` and `rir_ref.scp`; no audio generation
or duplication is required.

## Differences from v1

- Physical RIRs retain their original time coordinates. Both receive the SAME
  gain (division by the direct RIR's absolute peak); no independent peak shift
  or normalization is performed. In particular, no 2.5 ms training alignment.
- Full linear sweep/RIR convolution is retained, including the response after
  the sweep ends. Both waveform responses receive one FFT-window of zero guard
  on each side before centered STFT. Full frame convolution is retained too;
  the shorter spectrum is zero-padded for loss, never tail-truncated.
- Left-channel RIRs are padded to 32000 samples. Nonzero RIR content beyond this
  window raises an error rather than silently discarding the tail; increase
  `preprocessor_conf.rir_length` if needed.
- The WHAMR generator applies the same utterance gain to direct and reverberant
  speech, so this shared RIR normalization preserves their relative transfer.
- Model family/preprocessor: `rec_rir_sweep_v2`. Model architecture, input
  crop (4 s), input-only folded batching (batch size 4), validation batch size
  1, optimizer and scheduler remain those of the existing TF-Locoformer sweep
  configuration. No RIR-L1 auxiliary loss is used.

This changes both the target definition and the response window; v1 versus v2
is NOT an ablation of either change alone. Finite causal, band-diagonal CTF
approximation remains. Excitation-dependent weighting means that speech and
sweep objectives need not have identical optimal CTF coefficients.

## Run

From this recipe directory:

```bash
# Prepare/verify existing indexes and collect v2 statistics only:
bash run_tflocoformer_single_nf_16k_sweep_v2.sh --stop_stage 5
# Start scratch training after stats exist (on an authorized GPU host):
bash run_tflocoformer_single_nf_16k_sweep_v2.sh --stage 6 --stop_stage 6
```

Separate outputs:

- `exp/rir_stats_train_tflocoformer_single_nf_16k_sweep_v2_input_batch`
- `exp/rir_train_tflocoformer_single_nf_16k_sweep_v2_input_batch`

Existing v1 configs, checkpoints and experiment directories remain valid.
RIR inference uses the existing `espnet2.bin.rec_rir_inference` entry point and
PIM conversion. Its output represents the estimated direct-to-reverberant
transfer. Peak alignment/normalization for ACE/BUT evaluation does NOT convert
it into a clean-to-reverberant response; label those evaluations accordingly.

## Checks

The focused CPU test can run without pytest:

```bash
PYTHONPATH=../../.. python ../../../test/espnet2/rir/rec_rir/test_sweep_v2.py
```

It checks full waveform convolution against a known delayed response, late-tap
loss gradients, paired gain/colour preservation, penalties on excess predicted
tail, and rejection of nonzero RIR truncation. Real-data scratch update and
inference are also checked during implementation. The real-data CPU check used
the configured 4 s speech crop and full 8.192 s training sweep, with finite
gradients and an AdamW update (2,592,281 parameters). Only the inherited PIM
inference sweep was shortened to 0.1 s for the CPU smoke test; full-duration
GPU inference/performance has not been validated here. One-example ESPnet
train/valid statistics collection also passed. No long training is launched
as part of this change.
