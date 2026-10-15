# Estimated CTF temporal-concat conditioning (Small, 8 kHz)

This experiment uses the **same temporal-prefix separator as DAI-43**, replacing
its Oracle CTF with estimates from the fixed native-8-kHz BiMamba checkpoint.
It does not modify or resume the completed BiMamba FiLM experiment.

- Inputs: enh7_baseline's original NF-WHAMR 8-kHz observations and two direct
  speech references via `dump_nf_baseline_8k`; no audio conversion or duplication.
  Reuse the verified absolute SCP indexes in `exp/bimamba_nf8k_baseline_inputs`.
- TF-Locoformer Small: four blocks, embedding 96, STFT 256/hop 64. Batch 4,
  accum_grad 1, FP32, seed 0, four-second crops, AdamW and the same scheduler,
  max_epoch 150 and early stopping as the Baseline/Oracle configurations.
- BiMamba: `rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2/45epoch.pth`, SHA256
  `728afa1b8d27d526ba87cb73ca48193ee08f91ee48023cc1f26d9f44bd8a990e`.
  All **439,532** predictor parameters are frozen and remain in evaluation mode.
  It receives the actual mixture crop, never clean references or Oracle CTFs.
- Conditioning: the existing DAI-43 separator implementation is reused directly.
  A shared 1x1 real/imag projection (+288 trainable parameters) produces
  `[speaker-1 CTF:60][speaker-2 CTF:60][speech]` along the time axis. Training
  randomly exchanges speaker prefixes; the standard speech PIT loss is unchanged.
  Prompt frames are removed before the output convolution/iSTFT.
- CTF ordering follows the predictor's public `estimate_ctf` API: flip its raw
  training tap axis exactly once, then `[B,60,2,129,2]`. No extra tap reversal,
  amplitude normalization, or speaker-reference matching is introduced.
- The new config preserves CPU RNG around frozen-predictor construction, so
  same-seed Xavier initialization of the separator matches Oracle concat.
  This does not claim identical minibatch/random-augmentation trajectories.
  The established FiLM config/default construction behavior is unchanged.

Run from `enh_rir`, within an allocation:

```bash
bash run_nf8k_bimamba_ctf_concat_s.sh --stage 5 --stop_stage 5
bash run_nf8k_bimamba_ctf_concat_s.sh --stage 6 --stop_stage 6 --ngpu 1
# After completed training, use the same saved best checkpoint for both splits:
bash run_nf8k_bimamba_ctf_concat_s.sh --stage 7 --stop_stage 7 --ngpu 1
bash run_nf8k_bimamba_ctf_concat_s.sh --stage 8 --stop_stage 8 --ngpu 0
```

When reusing existing verified prepared indexes, skip Stage 5; paths inside
the indexes must remain valid. The default output is
`exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0`.

For the first TSUBAME run, create `qsub_logs` and submit
`qsub -g tga-shinoda qsub/train_nf8k_bimamba_ctf_concat_s_1gpu.sh`.
This uses `gpu_1=1`, priority -5, one hour. Before training it checks real
pretrained-Mamba forward/backward, unchanged frozen weights, and the longest
validation utterance on the allocated GPU. An OOM stops the run; no batch-size
or precision changes are made. Probe weights are discarded. For continuation,
override `h_rt` and pass the same allocation duration in seconds followed by
`true`; a saved checkpoint is then required.

CPU tests exercise axes/tap order, padding, PIT gradients, source preservation,
Oracle/Baseline configuration parity, same-seed separator initialization, and
both full and segmented inference. CUDA-only Mamba kernels are tested in-job.

For the node_q continuation check, use
`qsub -g tga-shinoda qsub/train_nf8k_bimamba_ctf_concat_s_nodeq.sh`.
This requests only `node_q=1` for 24 hours and requires the existing checkpoint;
it reuses the same GPU check and Stage 6 code without changing the model, batch,
precision, worker count, or optimizer. Allocation/GPU UUID/memory/process
snapshots are saved under `exp/bimamba_nf8k_concat_jobs/job_<ID>/`, and the CUDA
probe prints its own PID/namespace and free memory before building the model.
These are read-only diagnostics, not changes to GPU selection or isolation.
