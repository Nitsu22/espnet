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

## Pooled BiMamba Sweep v2 + DRR

`run_pooled_bimamba_2spk_nf_16k_sweep_v2_drr.sh` adds a DRR objective to the
full four-pre-block/two-post-block predictor. The network, parameter count,
input, teacher preprocessing, full-tail Sweep RIMag loss and ordinary-sweep
PIM inference remain the same. No speech or additional prediction head is
introduced. The shared Sweep v2 PIT implementation also accepts the optional
DRR settings for TF-Locoformer; their default weight is zero, preserving the
existing objective and checkpoint state dictionary.

For each predicted slot/reference pair, the predicted CTF filters that
reference's **direct sweep**. Differentiable iSTFT and the **original sweep's**
inverse then restore the clean-to-reverb basis for the auxiliary measurement.
The reference `STFT(sweep * rir_ref)` passes through the identical reconstruction
and inverse, avoiding disagreement caused solely by the finite-band sweep
measurement. The known inverse gain and delay are calibrated once; predicted
RIRs are not peak-aligned, peak-normalized or fitted to the reference.

DRR is `10 log10(E_direct / E_other)`, with a fixed inclusive +/-2.5 ms window
around the physical reference RIR's absolute peak. The same reference-defined
window is applied to both reconstructed responses; all samples outside it,
including pre-peak samples, form the denominator. Both energies use a small
total-energy-relative floor (1e-8), plus a float32 stability floor for silence.
This peak convention matches the evaluation's reference convention; it is not
an independently annotated direct-arrival time. The RIR measurement window is
32000 samples (2 seconds), as in evaluation.

One utterance-level PIT assignment minimizes the speaker-averaged combined
cost `loss_sweep + 0.1 * SmoothL1(DRR_pred - DRR_target)`, with a 1 dB Huber
transition. All reported terms use this SAME assignment. `loss_drr` is the
unweighted SmoothL1 term, and `drr_mae_db` measures this matched training
measurement, not the ordinary-PIM held-out evaluation metric. The weight 0.1
is an initial experimental choice, not a tuned or demonstrated improvement.
Early stopping and best checkpoint selection use `valid.loss_sweep`, keeping
the primary Sweep criterion; combined loss values should not be compared
numerically to Sweep-only runs.

The auxiliary clean-to-reverb reconstruction uses the known direct RIR only
when computing the loss. The network still receives only the mixture, and
ordinary PIM inference still returns a direct-to-reverb transfer without
restoring that direct path. This existing train/evaluation basis limitation
is retained for comparison with the baseline; the auxiliary measurement does
not remove it. Any improvement must be checked on the same WHAMR/BUT test sets.

After training, evaluate the primary-component checkpoint with:

```bash
python local/evaluate_two_speaker.py \
    --experiment exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_drr \
    --checkpoint valid.loss_sweep.best.pth \
    --data dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k \
    --output exp/pooled_bimamba_sweep_drr_whamr
```

Scoring in a separate stage must use the same `--checkpoint` value. Inference
and score alignment/normalization/PIT are otherwise unchanged. Unit tests in
`test/espnet2/rir/rec_rir/test_sweep_v2_drr_pit.py` check the energy definition,
gain/polarity invariance, silent-input gradients, matched reconstruction of a
non-impulse direct path, DRR-only CTF gradients, zero-weight compatibility and
shared assignment across the two losses.

## Portable dumps on TSUBAME

For Sweep + DRR training, submit
`qsub -g tga-shinoda qsub/pooled_bimamba_sweep_drr_train_1gpu.sh` from a dedicated
TSUBAME checkout's `rir_2spk` with a link to the verified portable NF dump.
The script uses `gpu_1=1`, priority -5 and a 24-hour limit, prepares input-only
Stage 5 shapes, validates a real batch-four update and full-length validation,
then runs Stage 6 from scratch with unchanged training settings. Its Mamba
wheel comes from the existing shared `damsep_clean/.deps` environment.
Logs and a profile report go to `exp/pooled_bimamba_sweep_drr_jobs/job_<ID>`.
If the time limit is reached, the script records `resume_ready`; an explicit
subsequent submission with argument `true` resumes the existing checkpoint.
Do not update the checkout while its job is queued or running.

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

### Completed-ablation evaluation on TSUBAME

`qsub/pooled_bimamba_ablation_infer.sh` evaluates the completed `2blocks` or
`bilstm` variant's `valid.loss.best.pth` (45epoch.pth) on the full 3000-example
WHAMR test set and BUT clean/noisy 16-kHz sets. It uses ordinary-sweep PIM,
full utterances, channel 0 and FLOAT output, matching the earlier comparison.
It requests `gpu_h=1` for inference only, with a real CUDA smoke check first.

Submit `qsub/pooled_bimamba_ablation_score.sh` with `-hold_jid` set to the
actual inference job ID. This `cpu_4=1` job requires successful inference
markers, applies the unchanged peak alignment/normalization and RMSE PIT,
and saves both ordinary and polarity-aligned BUT scores. The main BUT
comparison uses `score_polarity_aligned`, as for the earlier models.
`valid_metric_counts.json` reports finite source-pair counts alongside each
summary, and `evaluation_complete.json` marks all three conditions complete.

Run both scripts from an isolated evaluation checkout's `rir_2spk`, leaving
the ongoing training checkout intact. Arguments are the variant name and an
absolute result root. Models and dumps are read from the original TSUBAME
recipe; output metadata records the evaluation code commit and copied
checkpoint SHA256. GPU inference and CPU scoring can also be selected with
`local/evaluate_two_speaker.py --stage inference|score`; its default `all`
continues to perform the original combined workflow.

## 8 kHz pooled BiMamba baseline

`run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh` trains the four-pre-block,
two-post-block model from scratch on noise-free, two-speaker, channel-0 WHAMR.
Stage 1 builds zero-copy indexes in `dump_nf_2spk_8k_min` from the existing
8 kHz `data_rir_plus` manifests; it does not regenerate or copy audio.
The 4-second crop, batch 4, 60 CTF taps, sweep v2 objective and optimizer
match the 16 kHz baseline. FFT/window/hop are 256/256/128 samples (the same
32/32/16 ms durations); physical RIR teachers are 16,000 samples (2 seconds).
The frequency axis is 129 bins, so parameter count is 439,532 rather than
456,428 at 16 kHz. This is a sampling-rate comparison, not an exact replication
of DAMSEP's architecture or training settings.

```bash
bash run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh --stage 1 --stop_stage 1
python local/check_two_speaker_training.py \
  --config conf/tuning/train_pooled_bimamba_2spk_nf_8k_sweep_v2.yaml \
  --data-dir dump_nf_2spk_8k_min/raw/tr_rir_2spk_nf_min_8k \
  --output exp/bimamba_8k_gpu_check.json
bash run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh --stage 5 --stop_stage 6
```

Use an available CUDA GPU and a persistent session, with a writable
`NUMBA_CACHE_DIR`. Check the GPU smoke result before starting training.

## Independent architecture ablations (16 kHz)

The following four variants use the original four-pre/two-post-block baseline,
with exactly one requested architecture change. All share seed 0, batch 4,
4-second training crops, full-utterance validation with batch 1, 60 CTF taps,
SweepCTF v2 PIT, optimizer/scheduler, checkpoint selection and padding behavior.

| Suffix | Change | Parameters | Tensor GPU |
| --- | --- | ---: | ---: |
| `time2_freq4` | Time blocks at stages 1 and 3; all four lightweight frequency blocks | 309,100 | 4 |
| `time4_freq2` | All four time blocks; lightweight frequency blocks at stages 1 and 3 | 433,898 | 5 |
| `scalar_pool` | Speaker/frequency-specific scalar temporal weights shared across feature channels | 452,270 | 6 |
| `local_freq_only` | Remove the pre-pooling global frequency MLP branch; retain local depthwise convolution | 413,416 | 7 |

The baseline has 456,428 parameters. Depth selectors are zero-based `[0, 2]`
in YAML: the skipped branches are parameter-free identities in a fixed
four-stage schedule, preserving the order and depth of the other branch.
Thus the earlier `pre_layers: 2` experiment (both branches reduced) and the
new two experiments separate time and frequency contributions. The scalar
Pooling variant still has separate distributions for the two speaker slots;
it shares only the channel axis, not the speaker or frequency axes.
The global-MLP removal does not remove post-pooling frequency attention.

Each suffix has a YAML and `run_pooled_bimamba_2spk_nf_16k_sweep_v2_SUFFIX.sh`.
An isolated checkout may reuse the original recipe's prepared dump through an
absolute symlink. No audio copies are needed. On a verified free lab GPU:

```bash
bash local/run_pooled_bimamba_architecture_lab.sh \
  time2_freq4 4 /absolute/path/to/original/egs2/whamr/rir_2spk
```

Use a separate persistent session/log for each variant. The launcher validates
input-only shapes and then performs an actual CUDA update, checks every active
parameter's finite gradient, verifies teacher-permutation invariance, decodes
two RIRs and measures full-length validation (including the longest example).
It starts scratch training only after those checks pass. No dependency or
host configuration is modified. The launcher refuses an existing run marker
or checkpoint; it does not silently restart an experiment.

Models and stats are saved in the supplied result recipe's `exp/rir_train_...`
and `exp/rir_stats_...` directories. Provenance, GPU checks and exit codes are
under `exp/architecture_ablation_training_20261009/SUFFIX/`. A zero exit code
plus the completed training log/checkpoint is required to claim completion.
The baseline regression covers identical state keys, seeded initial weights
and forward outputs with default settings. CPU stand-ins verify topology and
objective wiring; actual Mamba execution is checked separately on the lab GPU.

## Post-pooling attention interaction ablations (16 kHz)

The four variants retain the four time/lightweight-frequency stages, weighted
speaker/channel pooling, slot embeddings and two post-pooling frequency blocks
of the 456,428-parameter baseline. They add one pre-normalized, four-head,
64-dimensional attention residual per post-pooling block. The added attention
has no separate FFN or convolution. The existing frequency block's FFN and
local convolution still operate separately per speaker.

| Suffix | Attention sequence per post-pooling block | Initial Tensor GPU |
| --- | --- | ---: |
| `post_speaker` | Frequency block, then attention across two speaker slots at each frequency | 4 |
| `post_joint` | Frequency block, then attention across all speaker/frequency tokens | 5 |
| `post_frequency` | Frequency block, then additional attention across frequencies within each speaker | 6 |
| `pre_speaker` | Attention across two speaker slots at each frequency, then frequency block | 7 |

Each attention projects the same feature dimension and uses the same number
of heads, giving all four variants 489,964 parameters: 33,536 more than the
456,428-parameter baseline. The CUDA check verifies this count before training.
Attention across frequencies uses frequency positions 0 through 256 for each
speaker, with the positions reset for the second speaker in joint attention.
The speaker-slot embeddings distinguish the slots; no fixed speaker ordering
is imposed on the paired teachers. The speaker-axis version shares its
attention weights across frequencies. These comparisons test interaction
scope, the order of speaker/frequency processing and the capacity-only
control. The default model remains identical to the pre-ablation baseline.

The only added predictor configuration entries are `post_interaction` and
`post_interaction_order`. Every variant shares seed 0, folded batch 4, four-second
training crops, full-utterance batch-one validation, the original padding
behavior, direct-sweep paired teachers, SweepCTF v2 loss/PIT, optimizer,
scheduler, early stopping and best-validation-loss checkpoint selection.
No speech-output or additional RIR loss is introduced.

Run each variant in its own persistent session from an isolated code checkout,
after checking the selected GPU is free:

```bash
bash local/run_pooled_bimamba_attention_lab.sh \
  post_speaker 4 /absolute/path/to/original/egs2/whamr/rir_2spk
```

`RIR_LAB_PYTHON` may select an existing compatible Python environment; the
launcher does not install packages or alter the host. It reuses the prepared
`dump_nf_2spk_16k_min` through the checkout's dump symlink, checks input-only
batching shapes, performs a real CUDA update and profiling, verifies all active
parameter gradients, paired-teacher PIT invariance, two-RIR inference and
full-length validation including the longest utterance, then starts Stage 6
from scratch. Existing model, stats or run markers cause a failure rather
than an overwrite or implicit resume.

After training exits successfully, the same pipeline selects whatever epoch
`valid.loss.best.pth` points to and automatically evaluates WHAMR only. It
first evaluates two test examples, infers the full 3,000-example test set on
GPU, releases CUDA for CPU scoring, and verifies all 6,000 scored source pairs
and CSV aggregates. Evaluation uses the existing ordinary-sweep PIM, full
utterances, channel 0, FLOAT RIR output, two-second RIRs, peak alignment,
independent peak normalization and full-RMSE PIT. BUT evaluation is omitted.
The reference-peak-to-50-ms error is `rmse_direct_50ms_mean`; the separate
`rmse_50ms_mean` measures the first 50 ms of each aligned RIR.

Training/profiling/provenance and pipeline phase/exit markers are under
`exp/attention_ablation_training_20261011/SUFFIX/`; models and stats are under
`exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_SUFFIX/` and
`exp/rir_stats_train_pooled_bimamba_2spk_nf_16k_sweep_v2_SUFFIX/` in the supplied
result recipe. Evaluation output is
`exp/attention_ablation_whamr_20261011/SUFFIX/whamr/`. Its sibling
`evaluation_complete.json` is written only after validating successful stage
markers, complete IDs/PIT pairs, unchanged scoring settings, checkpoint/config
hashes and matching CSV aggregates. A later-stage failure records the failed
phase and nonzero pipeline exit while preserving successful training markers.

## Frequency module replacements (2026-10-12)

The following four configurations retain four time/pre-frequency stages,
two post-frequency stages, the same two-slot pooling and shared CTF head.
Only `pre_frequency_module` and `post_frequency_module` differ from the
original 16 kHz SweepCTF v2 baseline. Extra post-pooling attention is disabled.

| Run suffix | Before pooling | After pooling | Total parameters |
| --- | --- | --- | ---: |
| `freq_pre_attention` | Frequency Attention + Conv-FFN | Existing Attention + Conv-FFN | 581,640 |
| `freq_pre_bimamba` | Frequency BiMamba + Conv-FFN | Existing Attention + Conv-FFN | 809,192 |
| `freq_post_bimamba` | Existing light frequency module | Frequency BiMamba + Conv-FFN | 570,204 |
| `freq_both_bimamba` | Frequency BiMamba + Conv-FFN | Frequency BiMamba + Conv-FFN | 922,968 |

The attention variant follows the full-frequency attention/FFN structure
of TF-Locoformer using this recipe's `FrequencyAttentionBlock`. Its lightweight
GLU/depthwise-convolution FFN differs from the original ConvSwiGLU. The
pre-attention and pre-Mamba variants use identical Conv-FFN definitions and
initial FFN weights. Post-Mamba retains the original post-block Conv-FFN and
its initial weights. The Mamba branches have separate forward/backward
parameters, reverse the frequency axis, and use d_state=16, d_conv=4 and the
existing expand=2 implementation. The time branches are unchanged BiMamba.

Before pooling, frequency processing has shape `[B*T, F, 64]`; after pooling
it has shape `[B*2, F, 64]`. All speaker slots share the post-frequency weights.
Mixers never run along the speaker axis. The pre-only/post-only and combined
experiments use identical initial replacement weights for their corresponding
stages. Independent CPU RNG streams preserve the baseline initialization of
the encoder, time modules, pooling, slot embeddings, head and retained FFNs.
Discarded initialization placeholders are absent from the parameter list.
Default switches preserve the existing checkpoint keys and exact outputs.

All four share seed 0, 4-second crops, folded batch 4, full-input batch-one
validation, 60 complex CTF taps, direct-sweep teachers, SweepCTF v2 RIMag/PIT,
the original padding behavior, optimizer, scheduler and early stopping. This
is a module-family ablation with different parameter counts; report size and
measured time alongside accuracy. Use full-length validation in CUDA checks
because pre-pooling frequency modeling processes every observation frame.

From a checkout with `dump_nf_2spk_16k_min` and the existing Python environment:

```bash
bash run_pooled_bimamba_2spk_nf_16k_sweep_v2_freq_pre_attention.sh \
  --stage 5 --stop_stage 6 --ngpu 1
```

The other suffixes use matching run wrappers/configurations and distinct model
and stats directories. Stage 5 creates input-only shapes; Stage 6 trains from
scratch unless resume is explicitly selected. No dataset audio is copied.
For WHAMR evaluation, report the physical clean-reverb reference and the fixed
`dump_ls_direct_reverb` relative-transfer reference separately; both use the
same ordinary-PIM model predictions. No comparison CSV is required.

CPU verification is `python test/espnet2/rir/rec_rir/test_pooled_bimamba_frequency.py`.
Its frequency-Mamba substitutes check layout and integration; real selective
scan is verified by `local/check_two_speaker_training.py` on CUDA before a run.

## Temporal local attention before BiMamba

`train_pooled_bimamba_2spk_nf_16k_sweep_v2_time_local_attention125.yaml`
adds one temporal attention block before each of the four existing time
BiMamba blocks. The frequency modules remain the original light Conv/MLP
before pooling and the two original frequency attention/Conv-FFN blocks
after pooling. Encoder, two-speaker pooling, CTF head, shared Sweep v2
direct/reverb teachers, full-tail loss and utterance PIT are unchanged.

```
Encoder
  -> (temporal local attention -> temporal BiMamba -> light frequency) x4
  -> two-slot weighted temporal pooling
  -> shared frequency attention/Conv-FFN x2
  -> complex CTF [B, 2, 257, 60]
```

Each local block uses LayerNorm, 64-dimensional features, four heads,
temporal RoPE, an output projection and residual addition. The window is
125 frames: 62 past frames, the current frame and 62 future frames. With
the unchanged 256-sample hop at 16 kHz, this is +/-992 ms. This is the
receptive field of each added block; the following BiMamba still processes
the full observation sequence. The input crop is at most four seconds,
and validation/inference retain the full observation. The CTF still has
60 taps, independently of the attention window.

The `TimeLocalAttention` implementation computes the exact symmetric
window in bounded query blocks and independent frequency-sequence groups.
It never creates a full long-input T-by-T attention matrix or mask. The
default execution chunks are 64 queries and 64 sequences; with window125,
each call uses at most 188 keys. Changing these chunks does not change
the attention window. Nonexistent positions beyond tensor edges are
excluded. Batch padding is intentionally treated like the existing
baseline, including in BiMamba/pooling; this is not a padding fix.

The configuration differs from the original baseline only by
`predictor_conf.time_local_attention_window: 125`. Setting it to zero
preserves legacy checkpoint keys and execution. Common initialization
and RNG state are preserved when enabling the addition. The model has
523,500 total parameters (523,484 trainable), including 67,072 new trainable
parameters across the four local attention blocks.

Run from this checkout's `egs2/whamr/rir_2spk` directory:

```bash
bash run_pooled_bimamba_2spk_nf_16k_sweep_v2_time_local_attention125.sh \
  --stage 5 --stop_stage 6 --ngpu 1
```

Select an available physical GPU via `CUDA_VISIBLE_DEVICES` using the lab
execution skill before running. For a worktree, use `--rir_exp` and
`--rir_stats_dir` to place training artifacts in the main recipe's `exp/`
when desired. Input shapes and output tags are distinct from the baseline.
For CPU verification, run
`python test/espnet2/rir/rec_rir/test_pooled_bimamba_time_local.py`; it checks
an independent dense-softmax output/gradient reference, exact locality,
chunk boundaries, legacy initialization, checkpoint loading and shared
Sweep v2 PIT. Before training, verify the full YAML and real Mamba kernels
with `local/check_two_speaker_training.py` on CUDA, including the longest
validation observation. GPU scratch-update weights are discarded.

WHAMR scoring should separately report the physical RIR and the fixed
LS direct-reverb reference using the same ordinary-sweep PIM predictions.
An extra comparison CSV is not needed.
