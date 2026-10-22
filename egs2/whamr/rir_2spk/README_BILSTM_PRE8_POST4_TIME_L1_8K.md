# BiLSTM pre8/post4 + time-domain Sweep L1, 8 kHz scratch training

Architecture: eight blocks of time BiLSTM + lightweight frequency processing,
two-speaker weighted temporal pooling, four shared frequency BiLSTM blocks,
then the CTF head. The predictor is imported unchanged from the original
pre8/post4 training commit `250eeb82fd9c0c071ef99e00cf6852d786c63b76`.

The existing SweepCTF v2 spectral PIT loss is augmented with the existing
**normalized time-domain Sweep L1, weight 1.0**. The prediction is the full
iSTFT of estimated CTF applied to direct-Sweep STFT; its target is the full
iSTFT of reverberant-Sweep STFT. Divide each waveform L1 error by the target's
L1 mass (clamped at 1e-8), then use one utterance-level PIT permutation for
the combined spectral and waveform cost. No inverse Sweep or predicted
peak alignment/normalization is used in this loss.

- Random initialization: `init_param: []`, `resume: false`, fresh optimizer.
- LR 0.001, AdamW, weight decay 0, no warm restarts, max 200 epochs.
- Halve LR after five consecutive epochs without validation improvement
  (`ReduceLROnPlateau.patience=4`); floor 1e-6.
- Select checkpoints, decay LR and early-stop using `valid.loss_sweep_min`,
  the spectral-only PIT minimum used for comparable model selection in the
  previous time-L1 experiment. Combined loss and both components are logged.
- Preserve the current 15-bad-epoch early stop (`patience=14` in ESPnet).
- Same NF two-speaker 8 kHz data, 4-second train crops, batch size 4,
  input-only folded batching, full-length validation batch size 1, seed 0,
  FFT 256 / hop 128, 60 CTF taps, 2-second RIRs, and no AMP.

```bash
bash local/run_bilstm_pre8_post4_time_l1_8k_lab.sh 1 \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
```

Results:
`exp/rir_train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4_time_l1_plateau_200ep`.
Control metadata:
`exp/bilstm_pre8_post4_time_l1_training_20261023`.

The launcher checks architecture parity, scratch initialization, scheduler
timing, manifest hashes and real GPU updates/PIT/RIR output. Profiling
updates are discarded before Stage 6. Numerical failure is not marked as
successful completion. The old pre8/post4 checkpoint is used only for a
CPU compatibility check, never as an initialization for this training.

Comparison with the old pre8/post4 result changes both the loss and the
training schedule. Isolating the time-domain loss effect requires a
pre8/post4 spectral-only run with this same new schedule.
