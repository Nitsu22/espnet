# Sweep response and inverse-filtered RIR objectives

Three independent scratch experiments share the same network/data/input batching:

| Experiment | sweep_loss_weight | rir_loss_weight |
|---|---:|---:|
| 1: existing sweep CTF response (RIMag) | 1 | 0 |
| 2: inverse-filtered RIR (L1) | 0 | 1 |
| 3: combined | 1 | 1 |

The combined weights are initial experimental settings, not a claim of balanced
contributions or optimal weights. `loss_sweep`, `loss_rir_l1`, and weighted total
`loss` are logged; inactive terms are zero. Existing configs default to 1/0
and retain the original objective and checkpoint parameter structure.

## Differentiable RIR path

Full frame convolution of STFT(sweep) and predicted CTF -> iSTFT -> linear FFT
convolution with the fixed inverse sweep -> remove known inverse-filter delay
`len(sweep)-1` -> fixed reference-length crop -> divide by absolute peak amplitude
(with an epsilon floor) -> sample-mean L1 against the prepared ground-truth RIR.

The full response spectrum is used for RIR recovery to retain the response tail.
The original sweep objective still uses its original excitation-length window.
No `no_grad`, detach, NumPy conversion, or predicted-peak index is used on the
prediction path. Training does not shift a predicted peak into place: the target
already has its peak at 40 samples (2.5 ms), and timing errors remain penalized.
Peak-amplitude normalization is differentiable almost everywhere and retains
polarity. Both reference and prediction have two seconds of support at 16 kHz.

A finite, band-limited sweep/inverse pair is not an exact delta inverse: even a
true-RIR response can retain inverse-filter coloration/sidelobes. The L1 target
is the prepared RIR itself, not a reference passed through the inverse filter.
This limitation must be considered when interpreting waveform-loss values.
Evaluation retains the existing peak-aligned PIM path for comparability; its
peak search is deliberately not copied into the training loss.

## Launchers

From this recipe, on an available GPU following the GPU skill:

```bash
# TF-Locoformer, experiment 2
bash run_tflocoformer_single_nf_16k_sweep_rir_l1.sh --stage 5 --stop_stage 6
# TF-Locoformer, experiment 3
bash run_tflocoformer_single_nf_16k_sweep_ctf_rir_l1.sh --stage 5 --stop_stage 6
# Rec-RIR counterparts
bash run_rec_rir_single_sweep_16k_rir_l1.sh --stage 5 --stop_stage 6
bash run_rec_rir_single_sweep_16k_ctf_rir_l1.sh --stage 5 --stop_stage 6
```

Each uses a separate config, experiment directory and statistics tag ending in
`_input_batch`; the shared dump is unchanged. Defaults remain FP32, maximum
batch 4, input-only folded threshold 160000, four-second training crops, and
full-utterance validation batch 1. Inference/evaluation uses the existing
`run_eval_rec_rir_single_nf_16k.sh` with the appropriate `--rir_exp`.
No new production jobs have been launched by this implementation change.

Validation: CPU network backpropagation for objectives 1/2/3 and custom weights;
fixed-delay/polarity and inverse FFT convolution versus SciPy; original sweep
objective regression; invalid-weight rejection; existing single-source tests;
configuration construction for all four new configs. The environment lacks
pytest, so the 13 numerical test cases were invoked directly (without installing
packages). GPU memory/throughput for the added loss has not been measured.
