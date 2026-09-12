# Small 1ch nocache TF-Locoformer with bidirectional temporal Gated DeltaNet

## Scope

This experiment replaces only `blocks.*.frame_path.attn` in the nocache
TF-Locoformer. Frequency MHSA (with uncached RoPE), both ConvSwiGLU FFNs,
normalization and STFT/iSTFT remain in the backbone. The temporal GDN has no
RoPE. This is an offline, bidirectional separator, not a streaming model.

- Separator: `tflocoformer_nocashe_bigdeltanet`
- Configuration: `conf/tuning/train_enh_tflocoformer_small_nocashe_bigdeltanet.yaml`
- Backbone: 4 blocks, embedding 96, 4 frequency attention heads, attention
  width 128, FFN hidden widths 128/128, FFN kernel 8, stride 1.
- Input: WHAMR! 8 kHz min, mono, noisy/reverberant mixtures; 2 clean sources.
- STFT: FFT/window 256 samples, hop 64 samples.
- Each temporal direction: 4 heads, key/query width 32 per head, value width
  64 per head, causal depthwise Q/K/V convolution with kernel 4 and SiLU.
- Forward and backward directions have independent parameters. Reverse the
  input before the backward convolutions/recurrence and restore time order
  afterwards. Concatenate both projected outputs and linearly project to 96.
- Trainable parameters: **5,811,876**. “Small” describes the backbone dimensions,
  not an equal-parameter comparison with the original Small model.

The existing Small training settings (including FP32, AdamW, SI-SNR PIT,
4-second segments and batch size 4) are retained. Standard AdamW weight decay
also applies to the new gate parameters; no FLA-specific optimizer grouping is
introduced. Old MHSA checkpoints cannot be loaded strictly into this model.

## Implementation source and backend

The original authors recommend FLA:
https://github.com/NVlabs/GatedDeltaNet#-frequently-asked-questions-faq

Reference implementation reviewed at FLA commit
`516143e31fce09925e6c39ac37148444bad176c4`:

- Layer: https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/layers/gated_deltanet.py
- Recurrence: https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/gated_delta_rule/naive.py
- Paper: https://arxiv.org/abs/2412.06464

This implementation uses **portable PyTorch recurrent and chunkwise backends** and
requires no FLA installation. The inspected FLA CUDA extra requires Torch >=2.7
and Triton >=3.3, whereas the existing environment uses Torch 2.1.0+cu118.
No environment upgrade or optimized CUDA backend was introduced.
The upstream MIT notice is preserved in
`espnet2/enh/layers/LICENSE.gated_deltanet`.

For unit-normalized q and k and a zero initial state:

```
log_alpha = -exp(A_log) * softplus(a_proj(x) + dt_bias)
beta = sigmoid(b_proj(x))
S_decay = exp(log_alpha) * S_previous
S = S_decay + beta * k * (v - k^T S_decay)^T
y = q^T S / sqrt(key_dim)
```

The layer then uses per-head RMS normalization, a SiLU output gate and an
output projection. The recurrence uses FP32 accumulation (FP64 for double
inputs). Every call starts fresh; no inference or convolution cache persists.
Unequal-length utterances are processed independently, including global
normalization/FFNs, so reversed padding cannot contaminate valid outputs.

`gdn_backend: recurrent` retains the original per-frame reference path.
The training YAML now selects `gdn_backend: chunk`: 32-frame chunks with batched
unit-lower-triangular solves and recurrence only between chunks. Both implement
the same gated delta rule; neither introduces parameters. Chunk sizes do not
restrict the temporal receptive field. No optimized FLA CUDA kernel is used.
Output and q/k/v/decay/beta gradients were compared at lengths 1, 7, 32, 33, 65,
including padding across chunk boundaries and strong forgetting.
GPU training is gated on a separate 3-minute 4-GPU real-data smoke job.

## Checks performed on midgar (2026-09-13)

Initial implementation checks used CPU with Torch 2.1.0+cu118.

```
NUMBA_CACHE_DIR=/tmp/numba-bigdeltanet OMP_NUM_THREADS=2 \
  /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python -m unittest \
  test/espnet2/enh/separator/test_tflocoformer_bigdeltanet.py -v
```

All 6 tests passed (including the added chunk/recurrent parity test):

1. Recurrence agrees with an independent matrix-transition expression and
   passes double-precision finite-difference gradient checking.
2. Future frames affect earlier outputs; swapping directions and reversing
   the fusion weights obeys time-reversal symmetry; calls do not leak state.
3. Decay initialization survives ESPnet initialization and gradients are finite.
4. Variable-length padding, explicit mono-channel inputs, state-dict loading,
   and invalid input rejection behave correctly.
5. The actual Small YAML builds through `EnhancementTask`; 1024-sample synthetic
   waveforms pass STFT, separation, SI-SNR PIT loss, backward, clipping and one
   AdamW step. The saved full model reloads through ESPnet and reconstructs
   finite waveforms with the requested length.

Additionally, comparison with the pinned FLA naive recurrence on FP32 tensors
(B=2, T=9, H=4, K=8, V=16) gave maximum absolute errors:
output **5.96e-8**, gradients of q/k/v/log_decay/beta **2.38e-7**.
This validates the recurrence, not optimized FLA-kernel or entire-layer parity.

The new launch script is `run_small_nocashe_bigdeltanet.sh`. It uses a separate
experiment directory, and does not hard-code a GPU device. It was syntax-checked
for launching training. No WHAMR! quality result is available.

## Four-GPU comparison protocol

Baseline: TSUBAME `enh1/exp/enh_train_enh_tflocoformer_nocashe_small_4gpu`.
Use the same raw dump and shape files (20,000 training, 5,000 validation,
3,000 test utterances), seed 0, global batch 4 split across 4 ranks, FP32,
AdamW lr=1e-3/weight_decay=1e-2, warmup=4,000 updates, gradient clipping=5,
no gradient accumulation, no dynamic mixing, 150 epochs maximum and validation
loss patience=10. A different parameter count (5.81 M vs 5.13 M) remains an
architectural difference; this is a matched-training-condition comparison.

`run_small_bigdeltanet_4gpu_smoke.sh` exercises the actual ESPnet launcher,
training, validation and checkpoint saving on eight 4–5 second real utterances
per split, for one epoch. It uses an isolated output and never resumes a
production checkpoint. The qsub script is prepared on midgar and remains
Git-ignored under `qsub/`. Production uses all utterances and a separate output.
For quality comparison, evaluate both systems using the same checkpoint policy;
do not compare one system's best checkpoint against the other's average.
