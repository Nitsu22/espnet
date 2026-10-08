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

Two no-post-frequency ablations have separate configs and entry points:

| Entry point suffix | Pre-pooling Time/Freq blocks | Post-pooling blocks | Parameters |
| --- | --- | --- | --- |
| `sweep_v2_no_postfreq.sh` | 4 | 0 | 371292 |
| `sweep_v2_no_postfreq_2blocks.sh` | 2 | 0 | 201434 |

The full entry point prefix is `run_pooled_bimamba_2spk_nf_16k_`.
Each removes the entire post-pooling Attention/FFN block stack, preserving
the pre-pooling lightweight frequency modules, two-slot/channel-wise pooling,
slot embeddings and shared CTF head. The two-block variant reduces both
pre-pooling Time and lightweight Freq module stacks together. All remaining
config values match the original pooled BiMamba Sweep v2 run, including seed,
teacher definition, PIT, padding behavior, batching and optimization. Each
uses its own experiment/stats directory and trains from scratch.

## Portable dumps on TSUBAME

`local/portable_rir_dump.py prepare` bundles every indexed WAV as immutable,
byte-identical audio (local hard links when possible), relocates SCP paths to
the destination and records SHA256 plus sample rate/channel/frame headers.
The NF dump retains both channels, all 20000/5000/3000 examples and all speech
and direct/reverb RIR teachers. The BUT dump retains its 8/16-kHz clean/noisy
sets, source/direct/reverb/noise audio and original/distributed RIRs. Original
midgar metadata is archived under `provenance/midgar`; the original NF source
tree recorded there is not a runtime dependency of a portable dump.

Copy into incoming directories with rsync before submitting
`qsub/verify_transferred_dumps.sh`. That CPU-only job verifies every audio
SHA256 and header, all SCP IDs/rates/lengths, then publishes both dumps.
`transfer_complete.json` records verified file counts and sizes. Stage 1
recognizes the verified portable NF dump and checks its metadata hashes;
Stage 5 uses the same original sample lengths to prepare batching shapes.
No resampling, quantization, RIR normalization or waveform regeneration is
performed by this transfer workflow.

## Controlled architecture ablations

Four additional configs and entry points isolate the proposed changes from
the full four-pre-block/two-post-block predictor. The full file prefix is
`run_pooled_bimamba_2spk_nf_16k_sweep_v2_`; each suffix also has its own
`conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2_<suffix>.yaml` and
experiment/statistics tag.

| Suffix | Architecture change from the full model |
| --- | --- |
| `ffn_only.sh` | Keep four pre blocks and two post-pooling frequency FFNs; remove only post-pooling Attention. This differs from `no_postfreq`, which removes the complete Attention/FFN stack. |
| `freq_before_pool.sh` | Keep the same two complete frequency Attention/FFN blocks and apply them immediately before temporal pooling, after the four lightweight Time/Freq blocks. |
| `2blocks.sh` | Use two pre-pooling Time/Freq blocks and keep both complete post-pooling frequency blocks. |
| `bilstm.sh` | Replace each of the four time BiMamba modules with a one-layer BiLSTM, input/hidden size 64 per direction, followed by a 128-to-64 projection. Keep all frequency modules and pooling unchanged. |

The before-pooling variant processes a frequency sequence for every frame
rather than for two pooled speaker slots, so it requires substantially more
activation memory and computation. Its frequency stack has the same parameter
count and layer definitions as the full model; placement, and the resulting
input features, change. BiLSTM changes the time model and its parameter count;
the hidden size is fixed rather than claimed to provide an exact parameter
match. Construction with the real Mamba backend gives these counts:

| Variant | Total parameters | Trainable parameters |
| --- | ---: | ---: |
| Full baseline | 456428 | 456412 |
| `ffn_only` | 422876 | 422876 |
| `freq_before_pool` | 456428 | 456412 |
| `2blocks` | 286570 | 286554 |
| `bilstm` | 461548 | 461532 |

The BiLSTM variant is approximately 1.1% larger than the full baseline.
Inference speed and peak memory still need measurement on the same hardware.

All four preserve the same NF-WHAMR mixture, left-channel input, four-second
crop, direct/reverberant RIR-pair preprocessing, full-tail Sweep v2 RIMag PIT
loss, shared two-speaker assignment, 60 CTF taps, optimizer, batch settings,
seed and evaluation code. Padding remains matched to TF-Locoformer Sweep v2:
there is no new sequence mask, including in the time modules or temporal
pooling. None of these variants adds a speech output or RIR-L1 loss.

On TSUBAME, collect input-only shapes for all four with the CPU-only script:

```bash
cd /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
qsub -g tga-shinoda qsub/pooled_bimamba_ablations_stage5.sh
```

For a single variant, the equivalent Stage 5 call is:

```bash
bash run_pooled_bimamba_2spk_nf_16k_sweep_v2_ffn_only.sh \
    --stage 5 --stop_stage 5 --ngpu 0 \
    --python /gs/bs/tga-shinoda/nitsu/anaconda3/envs/tf-locoformer/bin/python
```

Replace `ffn_only` with the other suffixes as needed. Stage 5 checks the dump
manifest and prepares each configuration's batching shapes; it does not train
the model. GPU training is a separate Stage 6 submission.

### One-GPU TSUBAME training

Charged profile job 8934455 validated all four scratch updates, paired-teacher
PIT invariance, two-RIR inference and full-length validation including its
longest utterance on H100. Median steady updates were 0.1600 s (`ffn_only`),
0.2087 s (`freq_before_pool`), 0.0867 s (`2blocks`), and 0.0863 s (`bilstm`).
Training peak allocations ranged from 4.32 to 10.27 GB. The successful Mamba
wheel is reused from `../damsep_clean/.deps/mamba-cu118-torch21-py310`, matching
TSUBAME's PyTorch 2.1/cu118 environment.

`local/launch_bimamba_ablations_tsubame.py` runs on midgar in a persistent
terminal session. It submits `qsub/pooled_bimamba_ablation_train_1gpu.sh` once
per variant with `gpu_1=1`, priority -5, Stage 6 only, and unchanged training
configs. It records every job ID, commit, command, time estimate and point
estimate in the requested state JSON. Estimates use 5109 folded training
batches and 5000 full-length batch-one validation examples per epoch, with
25% overhead plus 30 seconds per epoch. The old model's 55-epoch stopping
point is a runtime assumption, not a prediction of the new models' convergence.
Walltime sizing includes the configured 100-epoch case and is capped at the
published 24-hour limit: 24 hours for `ffn_only`/`freq_before_pool`, 20 hours
for `2blocks`/`bilstm`.

The training script stops two minutes before its hard limit and records a
restart marker only for timeout status 124 with an existing checkpoint.
The launcher waits for that job to leave the queue, then resumes the same
experiment with `--resume true` and sizes the next segment from remaining
epochs. Optimizer/scheduler and stopping criteria remain intact. Other
failures are recorded and are not silently retried. Keep the launcher alive
for these automatic continuations; its state is persisted after each change.
