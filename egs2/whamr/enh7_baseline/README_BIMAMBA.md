# TF-Locoformer-S temporal BiMamba ablation

This configuration matches `train_enh_tflocoformer_s_whamr_8k.yaml`: noisy
reverberant WHAMR, mono 8 kHz, anechoic two-speaker targets, four-second crops,
four blocks, embedding 96, FP32, batch 4, AdamW and PIT SI-SNR. Each block's
temporal MHSA is replaced by two independent official **Mamba-1** modules
(`mamba_ssm.Mamba`, state 16, causal convolution 4, expansion 2). The backward
module receives reversed time; its output is reversed back, concatenated with
the forward output, projected to 96 and passed through the existing residual.
Frequency attention/RoPE, both Macaron Conv-SwiGLU FFNs, normalization, STFT
and waveform output are retained. Temporal RoPE is removed with temporal MHSA.
No speaker split, RIR conditioning or new loss is added.

The separator has 5,547,636 parameters versus 5,125,252 in the baseline
(+422,384, approximately 8.24%). This is not a matched parameter count or a
measured speed improvement. Variable-length examples are processed
individually without padding to avoid reverse-state contamination. The original
baseline does not exclude padding from all its internal operations; compare
full-length or equal-length inputs when isolating the architectural effect.

`mamba_ssm` and matching CUDA extensions must be installed in the execution
environment. Imports are lazy so other enhancement models do not require Mamba.
The configuration selects `mamba_backend: cuda`. `torch` uses the official
reference selective scan with the same weights for CPU verification; it is
not intended for full training. `auto` chooses CUDA kernels on CUDA input and
the reference on CPU. Each call starts without cached state. An ESPnet
initialization hook preserves Mamba's dt projection, decay and skip parameters.

Run from `egs2/whamr/enh7_baseline` in an allocated GPU environment, reusing
the existing linked dump and statistics:

```bash
bash run_whamr_tflocoformer_bimamba_s.sh --stage 6 --stop_stage 6
bash run_whamr_tflocoformer_bimamba_s.sh --stage 7 --stop_stage 7
bash run_whamr_tflocoformer_bimamba_s.sh --stage 8 --stop_stage 8 --ngpu 0
```

The dedicated output is
`exp/enh_train_tflocoformer_bimamba_s_whamr_8k_1gpu_batch4`, including for CPU
scoring. Supply a separate `--enh_exp` when changing batch/GPU settings or seed.
The baseline checkpoint is not loaded; this is a scratch-training ablation.
These commands prepare no scheduler allocation and submit no job.

For TSUBAME, use an isolated checkout and reuse the original linked WHAMR/NF
dumps and WHAMR statistics. The existing Mamba wheel folder is supplied as an
argument, without changing the conda environment:

```bash
mkdir -p qsub_logs
qsub -g tga-shinoda qsub/train_tflocoformer_bimamba_s_1gpu.sh /absolute/path/to/mamba/dependencies
```

The job first verifies CUDA/reference output and gradient agreement, then
checks the original audio/teacher pairing, a real batch-four optimizer step
and the longest validation example. Failure stops the job before training.
The probe weights are discarded. Resources are `gpu_1=1`, priority -5 and
24 hours; the upper point estimate is `0.2 * 0.8 * 24 = 3.84`, excluding any
later continuation. Training resumes an existing checkpoint in this dedicated
experiment directory if a subsequent job is submitted.

## Implementation validation

The CPU checks use real Mamba parameters and the official reference scan.
They cover output/gradient agreement with the official Mamba forward using
that scan, reverse-time dependency, state isolation, preservation of special
initialization, padding isolation, waveform PIT training/optimizer step, and
checkpoint restoration. A separate CUDA-versus-reference output/gradient test
is included and skipped if no CUDA device is available. CPU reference checks
do not validate the fused CUDA kernels or separation performance.
