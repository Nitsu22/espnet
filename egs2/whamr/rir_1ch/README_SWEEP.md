# Rec-RIR with sweep-only supervision

This is a separate scratch experiment. The existing speech-supervised run is
not stopped or resumed by this launcher.

```bash
bash run_rec_rir_single_sweep_16k.sh --stage 1 --stop_stage 1
# On a free large-memory GPU (FP32 batch size 4):
bash run_rec_rir_single_sweep_16k.sh --stage 5 --stop_stage 6
```

Input: NF-WHAMR speaker 1, left microphone, 16 kHz min, at most four seconds
during training. Target: `rir_ref.scp` only. No direct or reverberant speech
reference is passed to the model. The shared dump remains unchanged.

The network retains Rec-RIR's encoder, both feature stacks, feature fusion,
CTF stack, attention pooling and CTF decoder. Speech and reverberant-spectrum
decoders are removed, including their parameters; no auxiliary speech loss
is computed. This is a network-output/objective variant, not just a loss-weight
ablation. It keeps 60 CTF taps and the existing inference conversion.

Reference RIR: choose left channel, locate the absolute peak on the original
full RIR, align it to sample 40 (2.5 ms), normalize by its absolute amplitude,
then crop/pad to 32000 samples. Polarity is retained. This reference preparation
is our experiment choice, not claimed as Rec-RIR's published training recipe.

Prediction: temporal frame convolution of `STFT(sweep)` with predicted CTF.
Target: `STFT(time-domain convolution(sweep, prepared ground-truth RIR))`.
Loss: mean absolute real error + imaginary error + magnitude error (`RIMag`).
Both sides are compared over the excitation-length window, following the
existing two-speaker sweep experiment; the convolution tail after that window
is excluded. The excitation is the existing 8.192-second logarithmic sweep
with its existing fades and 512-sample padding at each end. Target construction
does not backpropagate; prediction construction does.

Stats and checkpoints are isolated under `exp/rir_stats_train_rec_rir_single_sweep_16k`
and `exp/rir_train_rec_rir_single_sweep_16k`. Defaults: scratch, FP32, batch 4,
same optimizer/scheduler/epoch settings as the speech-supervised baseline.

Evaluation uses the common entry point, for example:

```bash
bash run_eval_rec_rir_single_nf_16k.sh \
  --rir_exp exp/rir_train_rec_rir_single_sweep_16k --dataset ace_clean
```

Use `whamr`, `ace_clean`, or `ace_noisy`. Inference needs only input speech.
The same channel, peak alignment/normalization, output length and metrics are
used for both models. Absolute gain and propagation delay are not scored.

Validation (2026-09-10): CPU checks passed for long-delay peak alignment,
SciPy time-convolution/STFT agreement, an impulse CTF, and convolution gradients.
ESPnet statistics collection passed on two train and two validation utterances,
producing both speech and RIR shape files. On aurum GPU 0 (RTX 3090), one real
four-second training example passed a scratch FP32 optimizer update with finite
gradients (loss 32.588703), followed by finite 32000-sample RIR inference.
The check also asserts that both speech decoder heads are absent. Report:
`exp/sweep_training_check/check.json`. These are smoke checks; full sweep training
has not been started and batch-size-four memory use is not validated on RTX 3090.
