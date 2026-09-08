# SMS-WSJ 1ch TF-GridNet training

Target: the single-DNN (DNN1), Wav+Mag+MC row of Table V in
https://zqwang7.github.io/publications/TASLP2023_TF-GridNet.pdf .
This is a reproduction using the supplied Roland dump, not a claim that the
published scores or all unpublished implementation details are reproduced.

## Training choices

- 4 TF-GridNet blocks, H=192, D=48, I=4, J=1, four attention heads;
  5,494,120 trainable parameters, one input channel, two output speakers.
- 8 kHz, 256-point STFT, hop 64, square-root Hann analysis/synthesis window.
- Equation (12): sum of per-speaker waveform and re-synthesized magnitude L1,
  utterance-level PIT, plus waveform and magnitude constraints between the
  summed estimates and summed direct-path targets. No RI loss, no DOA loss,
  no constraint against the noisy/reverberant observed mixture.
- Mixture standard-deviation normalization; identical scaling for targets.
- One randomly cropped segment (up to 32,000 samples) per utterance per epoch;
  validation uses full utterances. No dynamic mixing or augmentation.
- Adam, initial LR 0.001, plateau scheduler factor 0.5 and patience 3,
  gradient clipping L2 norm 1.0.
- User-approved fallback to Roland for paper-unspecified settings: seed 0,
  Xavier uniform initialization, Adam epsilon 1e-8, no weight decay,
  float32/no AMP, 100 maximum epochs, early-stop patience 5,
  five-best validation-loss checkpoint averaging.
- User approved random initialization (no Roland checkpoint initialization).
- tensor GPUs 0,1,2,3; global batch 4, one utterance per GPU, no accumulation.
  Fixed batches drop incomplete final batches: train 33,540/33,543 and
  validation 980/982 utterances per epoch. This is an implementation detail,
  not a paper-specified choice. Shorter-than-four-second utterances are retained.

## Data and protection of original files

`dump_roland_sms_wsj_1ch` contains copied metadata only. Its audio SCPs refer
via absolute paths to
`/net/fractal/work2/roland/research/sms_wsj_dump/dump_oracle3`.
Both mixtures and targets use microphone index 0. Original files are read only;
statistics, models, and logs are written under `exp_tfgridnet_sms_wsj_1ch_paper`.
All referenced files were checked for existence (33,543 train, 982 validation,
1,332 test utterances). Test data is not used for training/model selection.

## Run

From this directory, on tensor in tmux:

```bash
bash local/launch_tfgridnet_sms_wsj_1ch_paper_tensor.sh
```

The launcher refuses occupied GPUs, runs a real 4-second batch on each GPU with
DDP backward/optimizer/checkpoint verification, then runs Stage 5 and Stage 6.
The Conda environment is `/home/kslab/nitsu/.conda/envs/tf-locoformer`.
The working repository is shared with tensor, so source changes are made on
midgar and read through that same path; no remote source editing is performed.

Status: `exp_tfgridnet_sms_wsj_1ch_paper/run.status`.
Launcher log: `exp_tfgridnet_sms_wsj_1ch_paper/run.log`.
Training log: `exp_tfgridnet_sms_wsj_1ch_paper/tfgridnet_1ch_scratch/train.log`.

For a CPU formula, permutation-invariance, scale-invariance, sqrt-Hann
round-trip, gradient, and checkpoint check, run the matching `local/check_*`
script with `--cpu` and the launcher's Python environment variables.

## Local NVMe copy on tensor (2026-09-08)

The user requested a local disk copy after shared-storage reads were slow.
The copy runs in tmux `sms_wsj_copy_local` using
`local/copy_and_resume_tfgridnet_tensor.sh`. The original Stage 5 process tree
was stopped; no training checkpoint had been created.

- Destination: `/var/tmp/nitsu/tfgridnet_sms_wsj_1ch` on tensor's local XFS NVMe.
- All 107,571 referenced WAVs are copied unchanged (all six channels retained).
  Only the SCP paths in the new local dump change. Training still selects channel 0.
- Capacity is checked against exact source sizes with at least 50 GiB reserve.
- rsync performs transfer verification; every destination size is then checked.
  A `COPY_VERIFIED.json` marker is written only after successful verification.
- Copy status/logs: `exp_tfgridnet_sms_wsj_1ch_paper/local_copy/`.
- On success, the original attempt's logs/statistics are archived under
  `shared_read_attempt_before_local_copy`, and the training tmux is restarted
  with `TFGRIDNET_DUMP_DIR=/var/tmp/nitsu/tfgridnet_sms_wsj_1ch/dump`.
  GPU availability is checked before the smoke test and again before Stage 6.
- This path is host-local temporary storage; the shared source remains authoritative.
