# DAMSEP implementation check before NF-WHAMR training

Checked on 2026-10-06 against the [paper](https://arxiv.org/html/2609.29749v1)
and [official source at c37eeaade1c4e90ecde9901c95bcfb37c93be749](https://github.com/Wenanzhi/DAMSEP/tree/c37eeaade1c4e90ecde9901c95bcfb37c93be749).

The separation training network and objectives follow the paper and released
model. This is an NF-WHAMR adaptation, not a reproduction of the paper's
HETMIXR results or its complete distance-estimation evaluation.

| Component | Check |
| --- | --- |
| Network | Released six-block SPMamba, shared dereverberation/response branch, trainable scalar fusion and temporal softmax pooling retained. 7,184,128 parameters; 60 complex CTF taps per source/frequency. |
| Source assignment | Clean-waveform negative-SNR chooses one PIT permutation, shared with both auxiliary losses and CTFs, as in paper equations 4–6. The release defaults to fixed near/far order; NF-WHAMR has no such speaker-index guarantee. |
| Objectives | Clean waveform loss plus reverberant RI+Mag loss and reference-clean-to-reverberant CTF reconstruction. Weights 1 / 0.1 / 0.5; no direct RIR supervision. |
| Signal analyses | Separator Hann FFT/window 256, hop 64; response network Hann FFT/window 512, hop 128. Auxiliary losses retain the released FFT 512 / window 256 / hop 128 / square-root Hann. |
| Optimization | Adam lr 0.001, no weight decay, clip 5; plateau factor 0.5 and scheduler patience 5; maximum 500 epochs. |
| Early stopping | Fixed ESPnet's one-epoch offset: configuration patience 4 stops after five consecutive epochs without improvement, matching the release's Lightning patience 5. Scheduler patience is unchanged. |

## Numerical and implementation evidence

- Independent AST comparison with the pinned release found only the documented
  compatibility/shape fixes in the vendored computation. Details are in
  `espnet2/enh/damsep/vendor/UPSTREAM.md`.
- FP32 comparisons against the release gave maximum differences of 0 for
  negative SNR, 3.81e-6 for SI-SDR, and 8.34e-7 for complex frame convolution.
  Multi-example checks exercised batch/speaker alignment.
- Regression tests cover shared PIT, causal complex convolution, joint
  gradients, padding exclusion, linked data, native training/checkpoint
  resume/inference/scoring, and the five-epoch stopping behavior.
- Prior TSUBAME H100 CUDA check (job 8917383) passed a full six-block,
  four-second FP32 forward/backward/Adam update with finite gradients for
  every parameter. Peak allocated memory was 18.32 GiB.

## Deliberate experimental differences and scope

- Existing NF-WHAMR train/validation/test data and direct-path clean teachers
  replace HETMIXR. Matching reverberant teachers come from the linked existing
  source-image dump. Stage 5 checked all file headers/IDs/lengths and sampled
  mixture sums; no source dump is rewritten.
- Short utterances are retained; training crops all five signals together to
  at most four seconds. Validation and test use full utterances. The release
  excludes short examples and crops validation to four seconds.
- FP32 replaces the released runner's default mixed BF16. One GPU uses batch
  one and accumulation one; the release's per-device batch is also one, while
  its example configuration lists eight GPUs. The paper specifies no GPU
  count or global batch size.
- Validation uses clean SI-SDR with the two auxiliary losses, as in the release.
- Native CTF outputs are saved. Sine-sweep decoding to physical RIRs and DRR
  near/far decisions are outside this separation comparison; the official
  release also omits the paper's decoding pipeline.
- CTF reconstruction of a random mid-utterance crop lacks preceding clean
  context, an approximation inherited from the released training loader.
