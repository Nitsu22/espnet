# TF-Locoformer-S: clean speech and CTF reconstruction

`run_small_nocashe_ctf_joint_4gpu.sh` trains a new experiment with two waveform
SI-SNR losses, weighted **1:1**. The shared last TF-Locoformer block feeds the
existing clean-speech decoder and a new utterance-level CTF head. The CTF head
uses frequency-wise temporal softmax pooling and two linear layers to output
complex `[batch, 2 speakers, 129 frequencies, 120 taps]` filters.

Predicted clean speech is decoded, STFT-transformed again, and convolved over
frames with its predicted CTF. The reconstructed reverberant waveform is
compared to the corresponding speaker's noise-free reverberant reference.
Both branches and the shared blocks receive reconstruction gradients. One PIT
assignment minimizes the sum of the clean and reverberant losses; there is no
independent reassignment of filters or speakers. Neither speech teachers nor
oracle RIR/CTF are inference inputs. The ordinary enhancement inference runner
continues to output clean speech; the separator also returns `others['ctf']`.

## Comparison conditions

Baseline: TSUBAME `enh1/exp/enh_train_enh_tflocoformer_nocashe_small_4gpu`.
Its actual saved configuration defines the comparison:

- Existing `enh1/dump/raw/{tr,cv,tt}_mix_both_reverb_min_8k`, including noise;
  20,000/5,000/3,000 utterances; left microphone, 8 kHz.
- Same input and direct-path (`s1/s2_anechoic`) clean-reference WAVs, without
  regeneration or substitution by noise-free mixtures.
- Four blocks, embedding 96, two SwiGLU FFNs of width 128, RoPE without caching.
- FFT 256, Hann window, hop 64; four-second training crops, full validation.
- Seed 0, scratch Xavier initialization, FP32, global batch 4 over four ranks,
  original three folded shape files and 80,000-sample folding thresholds.
- AdamW lr 0.001, weight decay 0.01, clipping 5, no gradient accumulation,
  warmup 4,000 updates and ReduceLROnPlateau factor 0.5/patience 3; maximum
  150 epochs and early-stop patience 10.

Learning-rate scheduling, early stopping and the default checkpoint selection
monitor **clean-only PIT SI-SNR loss**, matching the baseline's single loss,
even when joint PIT chooses a different assignment. This separate metric does
not enter the training loss. The joint sum and each component are logged
separately. Added head parameters and the
additional training objective are intentional comparison differences.
Baseline inference logs use `valid.loss.best.pth`, rather than an averaged
checkpoint. The new default uses the corresponding `valid.loss_clean_pit.best.pth`
and the same ordinary scorer/output normalization.

At an 8-ms hop, 120 causal taps span approximately 0.96 s, matching the temporal
coverage of the existing 60-tap/16-ms-hop CTF predictor. Tap zero multiplies the
current frame. This finite, frequency-diagonal CTF is an approximation. Both
losses are scale invariant, so learned CTF gain is not uniquely constrained.

For a random crop starting mid-utterance, prior clean speech is unavailable.
The input remains the exact baseline four-second crop; the reverb loss excludes
the first `(120-1)*64+256 = 7872` samples (0.984 s). The clean loss uses the full
crop. Full-utterance examples starting at zero have no warmup exclusion. Padding
is excluded from pooling and losses. Reconstruction uses each utterance's actual
STFT boundary and only the interval observed by its WHAMR reference.

## Data preparation and running

From `egs2/whamr/enh_rir`:

```bash
# Verify source pairing and create indexes, without launching training.
bash run_small_nocashe_ctf_joint_4gpu.sh --stage 1 --stop_stage 1
# Train inside an allocated four-GPU job; GPUs are chosen by the caller.
bash run_small_nocashe_ctf_joint_4gpu.sh --stage 6 --stop_stage 6
# Continue this experiment's checkpoint.
bash run_small_nocashe_ctf_joint_4gpu.sh --stage 6 --stop_stage 6 --resume true
# Evaluate clean separation with the baseline's ordinary WHAMR scorer.
bash run_small_nocashe_ctf_joint_4gpu.sh --stage 7 --stop_stage 8
```

Defaults are `dump_ctf_joint` and the separate
`exp/enh_train_enh_tflocoformer_small_nocashe_ctf_joint_4gpu`. An existing dump is
not overwritten; stage 6 requires its successful preparation manifest. Fresh
training refuses an existing checkpoint unless `--resume true` is supplied.

The adapter reads original WHAMR indexes from `../enh1/data` (override with
`--source_data_root`). It verifies all utterance IDs, 8-kHz lengths, mixture
addition, and the original mixture/clean audio against the exact baseline.
Baseline PCM16 clipping and one quantization step are allowed when checking
identity. Baseline input/clean WAVs are never rewritten. Only the paired
`spk1_reverb.scp` and `spk2_reverb.scp` teachers are added to new indexes.
Original baseline shape files are retained, so additional supervision does not
change batch membership. Audio files are referenced without duplication.

On TSUBAME, the baseline noisy dump exists. Inspection on 2026-10-06 found no
original `enh1/data` or `enh_rir/data_rir_plus`, and no indexed speaker-specific
reverb teachers in existing dumps. The `enh_rir/dump_rir_clean` input is
noise-free and is not a replacement for the noisy baseline. Restore the matched
original speech/indexes before preparing there; the adapter fails instead of
silently accepting a different dataset. Midgar retains the original paired
`enh1/data` audio. Paths in the prepared indexes are absolute and must be
prepared for the host that will run training, rather than copying indexes with
midgar paths to TSUBAME.

## Verification

```bash
PYTHONPATH=../../.. python ../../../test/espnet2/enh/test_ctf_joint.py
```

Checks cover causal convolution and late-tap gradients, unchanged baseline
speech outputs with shared weights, padded CTF pooling, identity-filter waveform
reconstruction, joint PIT, both branches' gradients, an optimizer update,
checkpoint reload through ordinary enhancement inference, aligned preprocessing,
the production YAML, and refusal of mismatched data. CPU checks do not establish
four-GPU performance or trained separation quality.

Verification on 2026-10-06: all 10 tests passed. The production four-block YAML
passed a real 32,000-sample FP32 forward/backward/AdamW update with finite
gradients in both heads and all shared trainable parameters. The shell launcher
also completed one CPU epoch, validation and checkpoint saving using an isolated
small network and bounded real audio. Full paired-data verification on midgar
passed for all 20,000/5,000/3,000 utterances and produced `dump_ctf_joint` indexes;
baseline mixture/direct teachers and original batching shape files are retained.
The first utterance's three baseline WAV hashes in each split also matched
TSUBAME's copies. No production training or TSUBAME GPU check was launched.
