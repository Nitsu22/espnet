# Single-speaker TF-Locoformer: CTF and sweep supervision

Both configurations use the existing TF-Locoformer CTF predictor with one
output: two blocks, embedding dimension 96, four attention heads, attention
dimension 128, RoPE, and two SwiGLU convolutional FFNs of dimension 128.
There are no clean/reverberant speech prediction heads and no speaker PIT.
This follows the existing rir/ two-block architecture, adapted to 257 frequency
bins and one source. It is not parameter-count matched to BiSpatialNet.

Training data and optimization follow the corresponding single-speaker Rec-RIR
experiments: NF-WHAMR speaker 1, left microphone, 16 kHz, four-second training
segments, FP32, folded batch size 4, seed 0, and scratch initialization.
Both predict 60 complex CTF taps with FFT 512 and hop 256.

| Variant | Reference | Loss |
| --- | --- | --- |
| CTF | Direct-path and reverberant speech | RIMag between STFT direct speech convolved with estimated CTF and STFT reverberant speech |
| Sweep | Measured/generated RIR | RIMag between STFT sweep convolved with estimated CTF and STFT of time-domain sweep convolved with reference RIR |

The sweep variant reuses the single-source Rec-RIR sweep implementation:
8.192-second excitation, reference RIR peak aligned to sample 40, absolute-peak
normalized with polarity preserved, and cropped/padded to 32,000 samples.
The response is scored over the excitation-length window. These references
have different gain/delay conventions from the speech-supervised CTF variant;
the numerical training losses are not directly comparable.

From this recipe directory, run either experiment independently:

```bash
bash run_tflocoformer_single_nf_16k_ctf.sh
bash run_tflocoformer_single_nf_16k_sweep.sh
```

Do not run both on the same GPU without checking capacity. These scripts do
not select a GPU; set CUDA_VISIBLE_DEVICES in the execution environment.
They reuse prepared audio and create independent statistics/checkpoints:

- exp/rir_train_tflocoformer_single_nf_16k_ctf
- exp/rir_train_tflocoformer_single_nf_16k_sweep

Statistics use the matching exp/rir_stats_train_* directories.
Resume with --stage 6 --resume true; defaults start from scratch and refuse
an existing checkpoint. CTF-only auxiliary statistics are zero placeholders
because the network has no auxiliary heads.

The existing inference/evaluation runner loads the saved network configuration:

```bash
bash run_eval_rec_rir_single_nf_16k.sh --dataset ace_clean \
  --rir_exp exp/rir_train_tflocoformer_single_nf_16k_ctf
bash run_eval_rec_rir_single_nf_16k.sh --dataset ace_noisy \
  --rir_exp exp/rir_train_tflocoformer_single_nf_16k_sweep
```

Both dataset choices can be used with either model (also dataset whamr).
Evaluation uses the existing common RIR scoring protocol.

Verification on 2026-09-13: both configurations build through RIRTask and each
has 2,592,281 parameters. Four new single-source checks and six existing
2-source sweep/speech regression checks passed by direct test-function execution.
On shannon GPU 4 (A100 80GB), each variant passed a real 64,000-sample FP32
forward/backward step, finite-gradient checks, parameter update, and finite
32,000-sample RIR inference. Initial losses were 4.6647358 (CTF) and 13.0279474
(sweep); these are smoke-check values, not trained evaluation results.
Reports are exp/tflocoformer_single_checks/{ctf,sweep}/check.json (exit status 0).
Only bounded checks were run; full training has not been launched.
