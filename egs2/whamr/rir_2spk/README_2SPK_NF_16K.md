# Two-speaker NF-WHAMR at 16 kHz

Three scratch baselines use the left microphone of the same noise-free mixture
(`mix_clean_reverb = s1_reverb + s2_reverb`):

| Entry point | Network / supervision |
| --- | --- |
| `run_rec_rir_2spk_nf_16k.sh` | Existing `BiSpatialNetPIT`: direct speech, reverberant speech, and speech-CTF reconstruction (all three weights 1) |
| `run_tflocoformer_2spk_nf_16k_ctf.sh` | Existing basic two-output TF-Locoformer CTF predictor, speech reconstruction PIT |
| `run_tflocoformer_2spk_nf_16k_sweep_v2.sh` | Same TF-Locoformer predictor, full direct-sweep to reverberant-sweep PIT |

No slot-split, room/path or RT60 auxiliary variant is selected. Sweep v2 does
not include the optional RIR-L1 objective. Rec-RIR is the normal all-loss
baseline; its auxiliary speech outputs make it a method comparison, not a
loss-only comparison against the two CTF-only models.

## Data

`local/prepare_two_speaker_dump.py` creates `dump_nf_2spk_16k_min` indexes from
`data_rir_plus`, checks all audio/RIR headers and ID sets, and records manifest
hashes. Existing audio is referenced without duplication or modification.
The input is explicitly `mix_clean_reverb`, not the single-speaker input and
not the noisy `mix_both_reverb`. Counts: train 20000, valid 5000, test 3000.
Existing links to the single-speaker and ACE/BUT dumps are not overwritten.

Each utterance has `speech_direct1/2`, `speech_reverb1/2`, `rir_direct1/2` and
`rir_ref1/2`. Physical RIRs are stored independently of utterance gain; the
original generator applies the same gain to each speaker's direct and reverb
speech, preserving the corresponding direct-to-reverb transfer.

## Shared conditions and PIT

- Seed 0, FP32, batch size 4, validation batch size 1.
- Four-second training crop, common crop for mixture and all speech teachers.
- Input-only folded batching, speech fold length 160000 (same as 1ch runs).
- STFT 512 / hop 256, square-root Hann, 257 bins; 60 CTF taps per speaker.
- TF-Locoformer: 2 blocks, embedding 96, FFN 128; same trunk as 1ch.
- AdamW lr 0.001, grad clip 1, cosine warm restarts T_0=5000 optimizer steps;
  max 100 epochs, patience 10; same settings as the 1ch input-batch experiments.

For each utterance, PIT selects one of the TWO complete speaker assignments.
Rec-RIR uses the same assignment jointly across all three loss terms. Speech
CTF pairs each predicted filter with a reference speaker's direct and reverb
speech. Sweep v2 pairs each filter with BOTH that speaker's direct-sweep and
reverberant-sweep; swapping only the reverberant target would be incorrect.

Sweep v2 preserves full convolution tails and shared direct/reverb gain/timing.
Each RIR pair is scaled by its direct absolute peak, without peak alignment.
Nonzero RIR content exceeding 32000 samples is rejected. The recovered transfer
is direct-to-reverb; ordinary PIM inference does not restore the clean-to-direct
component. The inherited speech-CTF crop-boundary and CTF approximation limits
remain. These runs do not establish performance without held-out evaluation.

## Execution

Run an entry point with `--stop_stage 1` for indexes, `--stage 5 --stop_stage 5`
for model-specific statistics, and `--stage 6 --stop_stage 6` for training.
Default stages 1 through 6 prepare/verify indexes, collect stats and train.
Each entry point has its own experiment/statistics directory under `exp/`.

`local/check_two_speaker_training.py` checks the real noise-free mixture sum,
GPU FP32 scratch update, finite gradients, teacher-permutation invariance and
inference of two 32000-sample RIRs. The synthetic unit test is
`test/espnet2/rir/rec_rir/test_sweep_v2_pit.py` (runnable without pytest).

Stage 5 with input-only batching uses `local/prepare_input_shapes.py` to write
input shapes directly from header-verified `utt2num_samples`, checking the
preparation manifest hashes and all IDs. No feature means/variances are used
in these configurations. The resulting lengths were verified identical for
all 20000 train and 5000 valid utterances to ESPnet-collected lengths on the
corresponding WHAMR min examples. This avoids rereading all teacher waveforms
three times just to obtain the same batch partition. Full GPU update checks
including PIT permutation invariance and two-RIR inference passed for all three
models.

## Pooled BiMamba Sweep v2 predictor

`run_pooled_bimamba_2spk_nf_16k_sweep_v2.sh` selects
`pooled_bimamba_sweep_v2_pit`. Its config uses the same dump, preprocessor,
4-second crop, batch/fold lengths, optimizer, scheduler, stopping criteria,
60 CTF taps and full direct-sweep/reverb-sweep RIMag PIT loss as the
TF-Locoformer Sweep v2 baseline. The inherited loss averages the two speakers
and retains the complete response and prediction tails. Ordinary-sweep PIM,
CTF tap ordering and `local/evaluate_two_speaker.py` are reused unchanged.

The predictor is Conv2D(2 -> 64, 3x3), then four blocks of independent
forward/backward Mamba-1 (state 16, convolution 4, expand 2) plus lightweight
frequency processing (depthwise kernel 5 and 64 -> 16 channel compression,
257 -> 16 -> 257 frequency MLP shared across compressed channels). A
64 -> 32 -> 128 MLP provides two-slot, channel-specific temporal weights.
Softmax over time, a weighted sum and learned [2, 64] slot embeddings produce
the pooled representation. Two shared frequency blocks use 4-head Attention
with RoPE and a GLU/depthwise-convolution FFN, followed by the shared
LayerNorm/64 -> 128 -> 120 CTF head. Parameter count: 456428 total,
456412 trainable; the remaining 16 are fixed RoPE frequencies.

Padding follows the current baseline: full batched STFT, no sequence masking
in the predictor or pooling, and whole-sequence reversal in backward Mamba.
This is a matched-condition architecture comparison, not a padding fix.
Post-pooling blocks can be disabled with `predictor_conf.post_layers: 0` for
an ablation. The model has no speech heads, RIR L1 loss, or auxiliary losses.
Implementation checks are in
`test/espnet2/rir/rec_rir/test_pooled_bimamba_sweep_v2.py`; the recipe's existing
`local/check_two_speaker_training.py` also supports this model for a real GPU
update and two-RIR inference. Implementing this entry point does not start
a training run.
