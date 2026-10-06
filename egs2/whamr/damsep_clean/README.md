# DAMSEP on noise-free WHAMR, 8 kHz

This recipe was copied from `../enh4_clean`: `cmd.sh`, the common environment,
tool/script links, `enh.sh`, and `local/` data-preparation utilities. Existing
experiments, dumps, data links, run scripts, model configs and result documents
were not copied. The inherited `local/data.sh` can generate WHAMR data, but the
default runner does not call it: it reuses the baseline's existing audio.

The model is the released DAMSEP/SPMamba network, vendored with its licenses
under `espnet2/enh/damsep/vendor/`. See `UPSTREAM.md` there for the exact source
commit and small compatibility fixes. Training uses ESPnet rather than the
release's Lightning wrapper, with checkpoint resume and multi-GPU support.

## Data and comparison

The recipe uses two versioned relative symbolic links:

| Link | Target and use |
| --- | --- |
| `dump_clean` | `../enh4_clean/dump_clean`: NF-WHAMR mixture and both direct-path clean teachers |
| `dump_reverb` | `../enh_rir/dump_ctf_joint`: both matching isolated reverberant source images |

The original SCP files are read directly. `dump_clean` also makes their
existing recipe-relative WAV paths work from this directory. The noisy mixture
in `dump_reverb` is not an input: only `spk1_reverb.scp` and `spk2_reverb.scp`
are used. There is no additional data-generation stage, WAV copy, or rewritten
dump. Do not replace these links with new data directories.

Stage 5 validates all three splits (20,000/5,000/3,000 examples), rates,
channels, lengths and ID alignment. Sampled waveforms verify that the two
reverberant teachers sum to the existing noise-free mixture within PCM16
quantization error. The report and source-list hashes are stored under
`exp/damsep_stats_8k/data_check.json`. Original SCPs and WAVs are never edited.

Then standard ESPnet CPU statistics collection derives shape files for all
five waveform series on the training/validation sets. It uses full utterances,
without random cropping, model construction, Mamba, CUDA or feature whitening.
`model_conf.extract_feats_in_collect_stats: false` makes this work even before
Mamba is installed. Statistics outputs are separate from the linked dumps.

Unlike the released HETMIXR loader, this recipe keeps short utterances and the
complete baseline test set; it does not require/load RIR waveforms when direct
RIR supervision is disabled. Full validation/test utterances are used. Training
uses a common random crop of at most four seconds for all five waveforms.
Short utterances remain shorter; batch padding is excluded from the network
and every objective. Test data is not monitored during training.

## Architecture and losses

- SPMamba: six blocks, embedding 16, kernel 8, four attention heads; separator
  STFT FFT/window 256, hop 64, Hann. Mixture standard-deviation normalization
  and the released initialization are preserved.
- Shared BiSpatialNet: hidden 64, two speech layers, six CTF layers; its STFT
  FFT/window 512, hop 128, Hann. CTF output is complex `[B,2,257,60]`.
- Training clean-waveform loss: negative zero-mean SNR, weight 1. Validation
  uses negative SI-SDR, matching the released train/validation metric choice.
- Reverberant supervision: RI+Mag, weight 0.1. Reconstruction: reference clean
  spectra filtered with estimated CTFs vs. reverb references, RI+Mag, weight 0.5.
- The auxiliary spectral losses retain the release's FFT 512 / window 256 /
  hop 128 / square-root Hann. This differs from the CTF network's own STFT;
  changing it would be a separate experiment.
- Select one source permutation from the clean waveform objective, then apply
  it to both auxiliary losses and the CTFs. NF-WHAMR speaker indices do not
  imply near/far order; the release's `no_pit` distance-order assumption is
  therefore replaced. No direct RIR loss is added.
- CTF convolution is causal across frames, with correct batch/speaker axes.
  As in the release, random crops have no preceding clean-speech context;
  reconstruction at the start of a mid-utterance crop is an approximation.
- Adam lr 0.001, no weight decay, clip 5; ReduceLROnPlateau factor 0.5,
  patience 5; early-stop patience 5, maximum 500 epochs. FP32 is explicit
  (the released Lightning runner can default to mixed BF16 on CUDA).

## Run

Activate the existing `tf-locoformer` environment. The network requires the
released Mamba 1 API (`mamba-ssm==1.2.0.post1`), PyTorch with CUDA, `einops`,
`torch-complex` and normal ESPnet dependencies. Lightning is not required.
Do not change an existing environment's dependencies without checking them.

On TSUBAME, install matching official prebuilt wheels without changing the
shared conda environment:

```bash
python local/install_mamba.py
```

This requires the verified Python 3.10 / PyTorch 2.1+cu118 / C++ ABI=False
environment. It installs Mamba 1.2.0.post1, causal-conv1d 1.2.0.post2 and pinned
Python import dependencies into `.deps/mamba-cu118-torch21-py310`. `run.sh`
adds that directory to its Python path when present. The installer uses
`--no-deps` and never upgrades the active environment's PyTorch or packages;
download URLs and wheel SHA256 values are recorded in `installation.json`.
No CUDA compilation is performed on the login node.

`qsub/check_cuda_trial.sh` performs one synthetic four-second FP32
forward/backward/Adam compatibility check on a full GPU, without dataset
training or speed measurements. Submit without `-g`/`newgrp`, with no other
compatibility trial running. It requests one resource unit and at most three
minutes; CUDA compatibility results are saved to `exp/cuda_check/report.json`.
GPU timing/scaling experiments require a separate group-charged job.

Stages follow the usual enhancement recipe: **5 statistics (CPU), 6 training
(GPU), 7 inference (GPU), 8 scoring (CPU)**. Start from Stage 5 because the
formatted dumps already exist.

```bash
# In a CPU allocation: validate existing dumps and collect shapes.
bash run.sh --stage 5 --stop_stage 5 --ngpu 0 --nj 4
# TSUBAME CPU job; submit from this recipe directory.
mkdir -p qsub_logs
qsub -g tga-shinoda qsub/stats_cpu.sh
# After Stage 5 and Mamba setup, on an allocated CUDA host.
bash run.sh --stage 6 --stop_stage 6 --ngpu 1
# Resume on four allocated GPUs: global batch 4, per-rank batch 1.
bash run.sh --stage 6 --stop_stage 6 --ngpu 4
# Preserve effective global batch 4 on one GPU with gradient accumulation.
bash run.sh --stage 6 --stop_stage 6 --ngpu 1 --accum_grad 4
# Full test inference on a GPU, followed by a separate CPU scoring job.
bash run.sh --stage 7 --stop_stage 7 --ngpu 1
bash run.sh --stage 8 --stop_stage 8 --ngpu 0
```

`qsub/stats_cpu.sh` requests `cpu_4=1`, four statistics jobs with one thread
and no loader workers each, priority -5 and a 30-minute walltime. It uses the
existing tf-locoformer environment; no CUDA module or Mamba installation is
required for this CPU stage. Shape files are stored under
`exp/damsep_stats_8k/{train,valid}/` and consumed by Stage 6.

`run.sh` does not overwrite `CUDA_VISIBLE_DEVICES`. GPU host choice and TSUBAME
allocation/submission are separate. The CPU statistics job is independent
of GPU training.
Checkpoints/logs/config are under `exp/damsep_nf_8k`, with the default inference
checkpoint `valid.loss.best.pth`. Changing architecture or data requires a new
`--expdir`. To start a new run, choose a new output directory; default resume
preserves existing training state.

GPU count, per-rank batch and accumulation are distinct: `run.sh` keeps one
example per GPU, so effective global batch is `ngpu * accum_grad`. Decide that
experimental setting before training; switching GPU count without compensating
accumulation changes the optimizer's batch size.

Inference writes `enhanced_tt/{clean,reverb}/spk{1,2}.scp` and complex CTF NPZs
under `enhanced_tt/ctf/`. The default 0.9 peak normalization and PCM16 WAVs
match the baseline's inference path; CTFs retain their original scale. Source
order is consistent between both waveform outputs and CTFs. Both clean and
reverberant outputs are scored against their corresponding references.
`result_si_snr.txt` / `result_sdr.txt` contain absolute scores, not improvements.
Compare the **clean** scores to enh4_clean's TF-Locoformer-S results.

The saved CTFs describe transfer from the WHAMR direct-path reference to its
reverberant image. They are not decoded physical RIRs or distance estimates.
CTF-to-RIR sweep decoding and DRR/near-far evaluation are outside this recipe;
the released DAMSEP code also does not include its paper's decoding pipeline.

## Verification

Run model/loss/data tests from the repository root:

```bash
python -m pytest -q -o addopts= test/espnet2/enh/test_damsep.py
```

CPU tests use Mamba's official differentiable reference scan and PyTorch RMS
normalization, preserving weights and the network computation. They exercise
full network output/gradient paths, but do not validate CUDA/Triton kernels,
multi-GPU performance, memory consumption, or separation quality.
