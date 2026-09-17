# TF-Locoformer Sweep CTF v2 + Sweep RIR L1

This variant extends v2's direct-to-reverberant sweep objective:

```text
D = STFT(sweep * clean_to_direct_RIR)
R = STFT(sweep * clean_to_reverb_RIR)
P = D convolved over frames with predicted CTF
loss_sweep = RIMag(P, R)
recovered_RIR = inverse_filter(iSTFT(P)) with fixed gain/delay compensation
loss_rir_l1 = mean(abs(recovered_RIR - clean_to_reverb_RIR))
loss = sweep_loss_weight * loss_sweep + rir_loss_weight * loss_rir_l1
```

Both losses share the FULL predicted direct-sweep response. The RIR reference
uses the v2 preprocessor's common direct-peak scale, preserving the relative
physical gain and original timing. No predicted-peak alignment, peak amplitude
normalization, or detached RIR reconstruction is used.

The inverse is the ORIGINAL sweep's inverse (not an inverse of the direct
response). Its known central impulse gain is calibrated once from
`sum(sweep * flip(inverse))`. The fixed removal delay is `len(sweep)-1+n_fft`:
`n_fft` removes v2's left waveform guard. Centered STFT adds no further waveform
delay after iSTFT. FFT convolutions are linear, with sufficient zero padding.

A finite, band-limited sweep and its analytic inverse are not an exact delta
filter: some sidelobes/bandwidth error remains even for the true response.
Fixed scalar calibration corrects central gain, not this spectral error.
The L1 reference remains the raw, commonly scaled clean-to-reverberant RIR;
this approximation should be considered when interpreting absolute L1 values.

## Configuration and run

Both weights initially equal 1.0, matching the old combined-loss recipe's
starting convention. This does NOT guarantee equal gradient contributions.
`loss_sweep` and `loss_rir_l1` are logged separately. Tune weights on validation
data if needed. The default v2 model retains `rir_loss_weight: 0.0`; its loss
and state-dict compatibility are preserved (new buffers are nonpersistent).

```bash
bash run_tflocoformer_single_nf_16k_sweep_v2_ctf_rir_l1.sh --stop_stage 5
bash run_tflocoformer_single_nf_16k_sweep_v2_ctf_rir_l1.sh --stage 6 --stop_stage 6
```

The recipe uses the existing dump's `rir_direct.scp` and `rir_ref.scp`, and
separate `exp/rir_train_tflocoformer_single_nf_16k_sweep_v2_ctf_rir_l1_input_batch`
and corresponding `exp/rir_stats_...` directories. Architecture, batch/input
settings, optimizer and scheduler match pure v2; no new data generation is needed.

Ordinary inference still uses the base PIM and estimates the direct-to-reverberant
transfer. This training loss uses known clean-to-direct RIRs. It does not silently
turn ordinary inference into clean-to-reverberant estimation or add oracle
evaluation. Existing training jobs are not restarted by this addition.

## Verification

The focused tests in `test/espnet2/rir/rec_rir/test_sweep_v2.py` cover:

- v2 full-tail/paired-response behavior and truncation guards;
- inverse filtering versus independent SciPy waveform convolution;
- fixed delay, polarity and linear amplitude preservation;
- differentiability, weighted losses, RIR-only gradients and zero-weight v2 parity.

Run from the ESPnet root (pytest is optional):

```bash
PYTHONPATH=. python test/espnet2/rir/rec_rir/test_sweep_v2.py
```

A CPU smoke check using the full config and the first real WHAMR training
example passed a 4 s input / 8.192 s sweep forward/backward and AdamW update.
Initial losses were sweep 10.59736 and RIR L1 0.00462421 (seed 0); these scalar
magnitudes do not quantify relative gradient contributions. No full training
or evaluation is started by this implementation change.
