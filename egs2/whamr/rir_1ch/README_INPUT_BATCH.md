# Shared input-based batching for single-speaker Rec-RIR

Use `run_rec_rir_single_nf_16k_input_batch.sh` and
`run_rec_rir_single_sweep_16k_input_batch.sh` with stages 5–6 for new scratch
runs. Both use the existing 16 kHz NF single-speaker dump, left microphone,
maximum four-second random training crops, and full validation utterances.
Speech targets use the same crop as input; RIR targets retain their existing
alignment and normalization.

Only `speech_mix_shape` determines batches. Targets remain loaded for losses;
removing their shape files from the sampler does not remove supervision.
The folded threshold is 160000 samples (10 seconds). This matches the duration
of `enh1/enh.sh`'s default 800*100 samples at the WHAMR recipe's 8 kHz rate.
The threshold applies to original utterance lengths, as in that recipe, not
cropped lengths. Batch size 4 is a maximum, not a fixed batch size.
Validation uses batch size 1 to limit memory for full utterances; this is an
explicit validation-memory choice rather than an exact TF-Locoformer default.

The actual existing train shape files give identical UID batches in both
models: 20000 examples, 5109 batches, mean 3.91466 examples, range 2–4.
Validation gives identical 5000 single-example batches. Original runs used
10124 and 10856 train batches. Wall-clock speedup is not measured.

New experiment and statistics tags end in `_input_batch`. Existing launchers
retain their defaults for reproducibility, including the separately running
CTF-only ablation. Do not resume old checkpoints into this new experiment.
Optimizer and step-based scheduler settings are unchanged; equal batches now
provide equal update counts for the two models.

On 2026-09-13 the original NF and sweep jobs on shannon GPUs 0 and 1 were
stopped at user request. Last complete epochs were 41 and 40 respectively;
partial epochs 42 and 41 are not saved. Existing checkpoint/model/log files
and `exp/evaluation_snapshot_20260913` remain available. The separate
`rec_rir_ctf_only_16k_train` job was not a stop target.
