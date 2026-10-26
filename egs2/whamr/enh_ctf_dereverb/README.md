# TF-Locoformer-S with speaker CTF estimation and regularized deconvolution

This implements the initial Stage 1/Stage 2 architecture in
`ctf_dereverb_plan.md`: the ordinary four TF-Locoformer-S blocks feed separate
Direct and Reverb decoders, a speaker-shared bidirectional Mamba CTF estimator,
and optional differentiable complex CG deconvolution. The backbone retains
frequency-then-time attention. Neither Direct/Reverb teachers nor RIR metadata
are inference inputs. Refinement and Sweep supervision are deferred.

## Comparison configurations

| `--phase` | Configuration | Output used for the main Clean loss |
|---|---|---|
| `b1` | Standard TF-Locoformer-S, proposed Clean loss and crop policy | Direct estimate |
| `stage1` | Two decoders plus CTF reconstruction supervision | Clean prior S0 |
| `stage2` | Stage 1 plus CG, with joint fine-tuning | Deconvolved Clean |
| `stage2_nocg` | Stage 1 plus corresponding fine-tuning, without CG | Clean prior S0 |

Stage 1 and B1 start from random initialization with AdamW at 0.001. Both
Stage 2 configurations start from the same Stage 1 model weights and AdamW at
0.0001; this starts a new optimizer/scheduler, rather than resuming Stage 1's
optimizer. Use equal additional updates for the Stage 2 versus no-CG comparison.
The examples are single GPU, batch 4, FP32, seed 0, maximum 150 epochs,
warmup 4,000 updates, plateau LR scheduling, early-stop patience 10.
The initial gamma 0.1 and 8 CG iterations are hypotheses to validate.
The supplied configurations have 5,125,252 baseline parameters and 5,372,553
proposal parameters (about 4.8% more); runtime and memory still need measuring.

The same PIT assignment selected using S0 in Stage 1 or the final Clean output
in Stage 2 is applied to every branch. Clean losses combine waveform negative
SI-SNR with a linear complex-STFT spectral term, coefficient 1. Active component
gradient norms are logged every 100 training steps; B1 disables this extra
diagnostic. Reverb and
true-Direct-to-true-Reverb CTF spectral losses each have coefficient 1.
Stage 2 also keeps the S0 loss. B1 has only its Clean loss. A conventional
SI-SNR-only B0 remains in `../enh7_baseline` and must use the same data/teachers
for comparison. Training/validation loss determines the saved best checkpoint;
report final test SI-SDRi separately.

## Data definition and verified local source

The initial recipe uses noisy WHAMR `mix_both_reverb_min_8k`, 20,000 training,
5,000 validation and 3,000 test utterances, left microphone. Existing paired
indexes and the full audio verification manifest are in
`../enh_rir/dump_ctf_joint` in the original checkout. A new Git worktree does
not contain ignored audio/results; pass the original checkout's absolute
directory with `--paired_dump` when running from a worktree.

`../enh_rir/local/prepare_ctf_joint_dump.py` checks the exact baseline mixture
and Direct references against original WHAMR files, checks aligned sample
counts and speaker slots, and verifies `Y1 + Y2 + noise = mixture`. The paired
dump retains the baseline's actual mixture/Direct WAVs and references original
Reverb WAVs without rewriting them. `enh7_baseline/dump` and `enh1/dump` share
the original baseline audio; preparation therefore defaults to `../enh1`.
Existing shape files are retained. The prepared manifest records index hashes
and PCM16 quantization/clipping differences. In the existing local manifest,
the baseline has 2,650/1,350/646 clipped source samples for train/valid/test;
Direct/Reverb physical consistency at those samples is consequently imperfect.

The current WHAMR generator in
`../enh_rir/whamr_scripts/wham_room.py` creates its anechoic RIR using
`max_order=0`; this represents the microphone Direct path with propagation
delay/filtering, rather than the original dry WSJ waveform. The paired audio
generator applies each speaker's same spatial gain to its Direct and full
Reverb versions. These code definitions and verified pairings support the
initial recipe; original generator versions and independent physical RIR
metadata have not been audited by this implementation. Do not infer such
metadata from filenames or call an effective Direct-to-Reverb CTF the raw RIR.

The light startup check verifies manifest hashes, paired split IDs, shape
alignment/folding limits and one WAV's existence per index. It relies on the
original full preparation for waveform identity. Host-specific absolute audio
paths must be prepared again on another host; copying midgar indexes to
TSUBAME is insufficient. NF-WHAMR is not silently substituted.

## Crop, normalization and STFT

The target training interval remains 32,000 samples. Context covers the causal
CTF history plus centered STFT boundary windows, and a right STFT guard. Losses
and reported metrics use only the target interval; full validation uses the
whole utterance. The extra context increases actual network input duration and
changes training context relative to the original four-second B0. B1 uses the
same context policy as the proposed models to control this change.

For L=128 and hop 64, the minimum causal frame history is `(L-1)*64 = 8128`
samples, in addition to the 128-sample centered-window guard. The implementation
uses conservative 256-sample window guards, hop-aligned crops, and explicit
target offsets and valid lengths; utterance-edge and padded
frames are excluded appropriately. CTF reconstruction and the CG observation
mask use the same target/context boundary definition. Output waveforms are
cropped only after iSTFT of the extended predictions.

STFT/iSTFT use FFT/window 256, periodic Hann, hop 64, `center=true`, reflected
true-utterance boundaries, `normalized=false`, `onesided=true`, and explicit
waveform output lengths. There is no spectral power/log compression. Mixture,
Direct and Reverb use one scale derived from valid mixture samples. Padding
and added context never become evaluation targets. CG uses FP32/complex64,
causal linear convolution with its complex adjoint, and no circular division.

Fold thresholds are 200,000 samples for each of the original three shape
files, so the validated utterance lengths do not reduce batch size 4. The
additional two Reverb shape files are deliberately not added to batching.

## Commands

Activate an existing ESPnet environment first. The launcher never allocates or
submits GPU resources; on TSUBAME use the project's TSUBAME workflow first.
All paths below are relative to this recipe unless absolute. A new output
directory is used for every phase.

```bash
# Inspect validated data and the command without launching any training.
bash run.sh --phase stage1 --dry_run true --paired_dump /absolute/verified/dump_ctf_joint

# Stage 1: train all branches without CG.
bash run.sh --phase stage1 --paired_dump /absolute/verified/dump_ctf_joint

# Stage 2: first compare fixed Stage 1 models over validation gamma/iterations,
# then fine-tune. Supply model weights, not the trainer's checkpoint.pth.
bash run.sh --phase stage2 --paired_dump /absolute/verified/dump_ctf_joint \
    --init_param exp/enh_train_ctf_stage1_whamr_8k_1gpu_batch4/valid.loss.best.pth

# Matched no-CG fine-tuning, initialized from the same Stage 1 model.
bash run.sh --phase stage2_nocg --paired_dump /absolute/verified/dump_ctf_joint \
    --init_param exp/enh_train_ctf_stage1_whamr_8k_1gpu_batch4/valid.loss.best.pth

# Same Clean loss and input context, without auxiliary heads.
bash run.sh --phase b1 --paired_dump /absolute/verified/dump_ctf_joint

# Continue the selected phase's saved optimizer/model/iterator state.
bash run.sh --phase stage2 --resume true --paired_dump /absolute/verified/dump_ctf_joint

# Only if paired indexes need preparing on this host; refuses existing output.
bash run.sh --stage 1 --stop_stage 1 --paired_dump dump_paired \
    --baseline_root ../enh1 --source_data_root ../enh1/data
```

Evaluate the fixed Stage 1 model over validation before selecting Stage 2
settings. The dedicated runner predicts from mixture only and saves S0,
deconvolved Clean, final Clean and CTF separately:

```bash
python -m espnet2.bin.ctf_dereverb_inference \
    --train_config exp/enh_train_ctf_stage1_whamr_8k_1gpu_batch4/config.yaml \
    --model_file exp/enh_train_ctf_stage1_whamr_8k_1gpu_batch4/valid.loss.best.pth \
    --wav_scp /absolute/verified/dump_ctf_joint/raw/cv_mix_both_reverb_min_8k/wav.scp \
    --output_dir exp/stage1_valid_gamma01_iter8 --device cuda \
    --enable_deconvolution true --gamma 0.1 --iterations 8 \
    --ref1_scp /absolute/verified/dump_ctf_joint/raw/cv_mix_both_reverb_min_8k/spk1.scp \
    --ref2_scp /absolute/verified/dump_ctf_joint/raw/cv_mix_both_reverb_min_8k/spk2.scp
```

The two reference indexes enable SI-SDRi and scale-dependent SNR scoring with
the same selected speaker assignment for all outputs; they are not model
inputs. Mono files and the first (left) channel of stereo WHAMR files are read,
matching training. Use `--device cpu` for CPU inference. STOI needs its optional
dependency.
Use a separate output directory per gamma/iteration pair. After Stage 2,
evaluate the test set with that phase's config/weights and the test indexes;
select settings using validation only.

`--phase stage2` and `stage2_nocg` require `--init_param` unless resuming; an
explicit `--allow_scratch true` enables the from-scratch ablation. Existing
checkpoints require `--resume true`. Configuration overrides use `--enh_config`;
keep a separate `--enh_exp` for changed settings. Temporary files use a newly
created directory under the user's `/tmp/nitsu`, cleaned at exit; compiler and
plotting caches stay under this recipe's `.cache`.

## Validation still required before research conclusions

From the repository root, the following runs the new operator/model/inference
checks and the existing joint-CTF regression checks (42 tests). Use an existing
ESPnet environment with PyTorch and official `mamba_ssm` installed; the small
models use Mamba's CPU reference scan. Set `TMPDIR` and numerical-library caches
to an owned temporary directory before running.

```bash
python -m unittest \
    test.espnet2.enh.layers.test_ctf_deconvolution \
    test.espnet2.enh.layers.test_speaker_ctf_estimator \
    test.espnet2.enh.test_ctf_dereverb \
    test.espnet2.enh.test_ctf_dereverb_inference \
    test.espnet2.enh.test_ctf_joint
```

Local unit/integration tests establish operator, gradient, masking and PIT
behavior, rather than trained speech quality. A tiny synthetic CPU trainer
smoke check also completed one train/validation minibatch in each stage,
loaded Stage 1 weights into Stage 2, and saved optimizer state plus best-loss
and best-SI-SNR checkpoints in both stages. No production training, GPU
throughput measurement, independent-probe reference-CTF audit, unseen-RIR test,
Refinement experiment or Sweep experiment is claimed by this recipe.
Before production Stage 1, verify Direct/Reverb definitions for the intended
corpus and characterize effective-CTF fitting error with independent probes.
Before production Stage 2, evaluate fixed Stage 1 predictions on validation
using the planned gamma/iteration candidates. Score S0 and deconvolved output
separately, use the same teachers/PIT policy, and record wall time and memory.
