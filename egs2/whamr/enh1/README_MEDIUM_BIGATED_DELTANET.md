# Medium 1-GPU bidirectional Gated DeltaNet experiment

This scales the Small nocache GDN experiment to the TF-Locoformer Medium
backbone: 6 blocks, embedding width 128, and ConvSwiGLU widths 192/192.
Only temporal MHSA is replaced. Frequency MHSA, normalization, FFNs, and
STFT/iSTFT remain unchanged. Each independent GDN direction retains 4 heads,
query/key width 32 per head, value width 64 per head, and convolution size 4.
The backend is the portable PyTorch chunk implementation, not optimized FLA.
This is offline bidirectional separation, not streaming.

Total trainable parameters: **16,380,404**, compared with approximately
14.97 M in the original Medium backbone.

Configuration: `conf/tuning/train_enh_tflocoformer_medium_nocashe_bigdeltanet_1gpu.yaml`.
Launch: `run_medium_nocashe_bigdeltanet_1gpu.sh` (Stage 6 only, resumable).
Output: `exp/enh_train_enh_tflocoformer_medium_nocashe_bigdeltanet_1gpu`.

Training uses the same WHAMR! mono 8 kHz min noisy/reverberant mixtures and
two clean speech references as the Small GDN experiment. FP32, seed 0,
4-second training segments, SI-SNR PIT, AdamW lr=1e-3/weight_decay=1e-2,
4,000-update warmup, gradient clipping 5, maximum 150 epochs and validation
patience 10 are retained. One GPU uses batch size 1 with four-step gradient
accumulation, keeping an effective batch of four. ESPnet advances the warmup
scheduler after optimizer updates. Validation uses batch size 1 and full
utterances. Accumulation preserves the effective batch size but does not
guarantee identical minibatch ordering or floating-point results to 4-GPU DDP.

The baseline Medium nocache run is `exp/enh_train_enh_tflocoformer_nocashe_repro_4gpu`
on TSUBAME. It uses 4-GPU DDP and additionally reads noise references/shape
files; its configured loss is still speech SI-SNR PIT. Compare with the same
checkpoint selection policy and report parameter counts and execution setup.
Medium here describes the shared backbone, not matched total parameters.
Small weights are not resumed into the Medium model; training starts fresh.

Before production, run `local/check_medium_bigdeltanet_gpu.py` on the selected
GPU to check four accumulated real 4-second microbatches, finite gradients,
one AdamW update and the longest full validation utterance. Then run
`run_medium_bigdeltanet_1gpu_smoke.sh` to check the ESPnet launcher, accumulation,
validation and checkpoint saving on eight real utterances per split.
The smoke uses a separate output and never resumes production checkpoints.

CPU checks:

```bash
NUMBA_CACHE_DIR=/tmp/numba-deltanet-medium OMP_NUM_THREADS=2 \
  /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python -m unittest \
  test.espnet2.enh.separator.test_tflocoformer_bigdeltanet -v
```

The seven tests include Medium waveform loss/backward/AdamW and full-model
save/reload. GPU runtime and memory must be measured before extrapolating
training duration. No quality or SOTA claim is implied by the launch.
