# Single-speaker model comparison

All variants use the same NF-WHAMR speaker-1/left-channel dump at 16 kHz,
maximum four-second training crops, FFT 512/hop 256, and 60 CTF taps.

| Variant | Parameters | Speech heads | Objective |
|---|---:|---|---|
| Original Rec-RIR | 3,138,144 | Yes | Direct/reverberant speech estimation plus speech reconstruction |
| Rec-RIR CTF | 3,119,132 | No | Direct-to-reverberant speech reconstruction |
| Rec-RIR sweep | 3,119,132 | No | Sweep response reconstruction |
| TF-Locoformer CTF | 2,592,281 | No | Direct-to-reverberant speech reconstruction |
| TF-Locoformer sweep | 2,592,281 | No | Sweep response reconstruction |

Counts were measured by building all five current configurations on CPU.
CTF and sweep have identical state-dict keys/shapes within each architecture;
strict loading between each pair passed. Rec-RIR uses the shared
`espnet2/rir/rec_rir/ctf_only.py` class. Original Rec-RIR retains its original
heads and default network selection, including compatibility with saved configs.
The old CTF-only experiment retained speech heads; its saved config still builds
that original architecture. Do not load it into the new headless configuration.

Use these launchers for the four controlled variants:

- `run_rec_rir_single_nf_16k_ctf_only.sh`
- `run_rec_rir_single_sweep_16k_input_batch.sh`
- `run_tflocoformer_single_nf_16k_ctf.sh`
- `run_tflocoformer_single_nf_16k_sweep.sh`

All select input-only folded batching with threshold 160000 samples, maximum
batch size 4, and validation batch size 1. Their new experiment tags end in
`_input_batch`. Start from scratch; old checkpoints/results remain unchanged.
Original Rec-RIR can use `run_rec_rir_single_nf_16k_input_batch.sh` as a baseline.
Launcher argument-capture checks passed for all four controlled variants.

## What the comparison establishes

CTF versus sweep changes the teacher definition as well as the excitation/loss:
CTF learns the transfer from direct-path speech to reverberant speech; sweep
uses an independently peak-normalized reference RIR aligned to sample 40 and
cropped/padded to two seconds. This is not a loss-only ablation. Evaluation
applies a shared alignment/normalization protocol but does not remove possible
spectral differences in the learning targets.

Compare architectures within CTF, and separately within sweep. The two
architectures have different parameter counts; report these alongside timing,
not as capacity-matched models. Original Rec-RIR is an auxiliary-supervision
baseline. Cropping-boundary reconstruction mismatch and network processing of
padded samples are unchanged by this refactoring.

## Timing protocol

`local/benchmark_single_rir.py` measures synchronized, warmed-up FP32 training
updates (forward, objective, backward, clipping, AdamW) and single-example
four-second inference including CTF-to-RIR conversion. It reports mean/median/
standard-deviation latency, peak allocated/reserved GPU memory, parameter counts,
input IDs/hash, config hash, and GPU/PyTorch version. Disk reading and preprocessing
are excluded. Repeated iterations reuse a fixed real batch; this is a compute
benchmark, not end-to-end training throughput or a quality evaluation.

```bash
PYTHONPATH=../../.. OMP_NUM_THREADS=1 \
NUMBA_CACHE_DIR=/tmp/nitsu-rir-benchmark \
/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python local/benchmark_single_rir.py \
  --config conf/tuning/train_rec_rir_single_nf_16k_ctf_only.yaml \
  --data-dir dump_nf_16k_min/raw/tr_rir_single_nf_min_16k \
  --batch-size 4 --warmup 3 --iterations 10 --device cuda:0 \
  --output exp/benchmark_single_rir/rec_rir_ctf.json
```

Run all configurations sequentially on the same otherwise idle GPU, environment,
inputs and batch size following the lab GPU skill. Compare matching input hashes.
GPU timing has not been run for this change; no latency/speedup numbers are claimed.
No new production training was started. Python compilation, shell syntax,
configuration/model construction, structural parity and launcher checks passed.
