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

From this directory, `local/prepare_nf_dump.py` creates
`dump_nf_8k_min/raw/{tr,cv,tt}_mix_clean_reverb_min_8k`. The lists reference
existing audio without copying or regenerating WAVs:

| Input/teacher | Source |
| --- | --- |
| Mixture | `../enh4_clean/dump_clean/raw/*/wav.scp` |
| Two direct-path clean references | Same dump's `spk1.scp`, `spk2.scp` |
| Two reverberant source images | `../enh4_clean/data/*/spk1_reverb.scp`, `spk2_reverb.scp`, resolved against `../enh1/data/whamr/2speakers/wav8k/min` |

These are the baseline's exact utterance IDs, formatted mixture and clean
reference WAVs. All channels/rates/lengths are checked; training takes the
left channel, 8 kHz. A sample spread through each split also verifies mixture
summation and clean-reference pairing/gain within PCM16 quantization error.
`preparation.json` records source-list and output-list hashes and check coverage.
Use `--waveform-checks 20000` for waveform checks on every example in every split.
Paths in output SCPs are relative to this recipe, for shared layout on another
host. The old absolute paths in the reverb lists are not used as-is.

TSUBAME already has the matching source images under
`../enh_rir/dump_ctf_joint`, while its original `data/` directories are absent.
Use `--reverb_dump ../enh_rir/dump_ctf_joint` for Stage 1 there. Only the two
isolated reverberant sources are read from that dump; the mixture and clean
teachers remain the exact NF-WHAMR baseline files. ID, audio-header and
mixture-summation checks still apply. In this mode, a comparison of clean
references to removed original raw files cannot run; `max_clean_error` is null.

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

```bash
# Only build/verify data indexes; no training.
bash run.sh --stage 1 --stop_stage 1
# TSUBAME: prepare indexes in a CPU allocation, using existing reverb teachers.
bash run.sh --stage 1 --stop_stage 1 --reverb_dump ../enh_rir/dump_ctf_joint
# On an allocated CUDA host. Default: one GPU, one example per GPU.
bash run.sh --stage 2 --stop_stage 2 --ngpu 1
# Resume on four allocated GPUs: global batch is 4, per-rank batch is 1.
bash run.sh --stage 2 --stop_stage 2 --ngpu 4
# Full test inference, then the same CPU scorer used by enh4_clean.
bash run.sh --stage 3 --stop_stage 4 --ngpu 1
```

`run.sh` does not overwrite `CUDA_VISIBLE_DEVICES`. GPU host choice and TSUBAME
allocation/submission are separate; no production job is submitted by setup.
Checkpoints/logs/config are under `exp/damsep_nf_8k`, with the default inference
checkpoint `valid.loss.best.pth`. Changing architecture or data requires a new
`--expdir`. To start a new run, choose a new output directory; default resume
preserves existing training state.

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
