# NF-WHAMR: frozen Rec-RIR CTF temporal prompts with zero Glue

Use the exact conditioning implementation from
[DAI-117](https://linear.app/nitsu22/issue/DAI-117), replacing Oracle CTFs with
estimates from the original frozen native-8-kHz two-speaker Rec-RIR.
The dataset remains NF-WHAMR, with the original `enh7_baseline` mixture and
anechoic teacher WAVs, mono channel 0, seed 0, FP32, one GPU, physical batch 4,
accumulation 1, four-second crops, full-length batch-one validation and the
unchanged TF-Locoformer-S optimizer, scheduler and stopping rules.
DAI-117 uses noisy WHAMR; only its conditioning method is reused here.

```text
[speaker-1 estimated CTF:60][zero Glue:4]
[speaker-2 estimated CTF:60][zero Glue:4][speech:T]
```

The separator source is identical to DAI-117 commit `1d5e90d019`:
`espnet2/enh/separator/tflocoformer_separator_nocashe_ctf_concat.py`, SHA256
`6d711435c80beaf01fc54b9a915293e60bb20cf5e9e20226701ed9ef5380b573`.
Both speakers share the 1x1 real/imaginary projection. Four exact zero feature
tokens are inserted AFTER each projected CTF; they are visible to every block.
Training randomly swaps the two speaker prefixes. The 128 prefix frames are
removed before the output convolution. A four-second training observation has
501 speech frames and 629 total frames. Glue adds no learned parameters; the
prompt projection adds 288 relative to the baseline. The separator's same-seed
Xavier weights exactly match the Oracle Glue model before training.

CTFs are estimated from precisely the separator's observation/crop, without
references or other utterances' padding. The frozen estimator and pinned source
are the same as [the Rec-RIR experiment](RECRIR_CTF_NF8K.md): original best
`97epoch.pth`, SHA256
`5fe27c7996e70d290d8646fb39f9cb6fe6a3f2972138e5f16b42730c8f4cafc8`.
Its public `estimate_ctf` API handles normalization and causal tap order.
No FiLM blocks are used by this separator. The stopped FiLM experiment's output
directory and pipeline markers are not reused.

Run from a dedicated, synchronized TSUBAME checkout's `enh_rir` recipe, with the
same read-only baseline dump/index links and pinned estimator asset:

```bash
python local/check_recrir_readiness.py \
  --config conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_concat_glue.yaml
qsub -g tga-shinoda -p -5 -l cpu_4=1 -l h_rt=00:10:00 \
  -N recrir_glue_control qsub/nf8k_recrir_pipeline.sh \
  --variant concat_glue --phase control
```

The initial GPU allocation first verifies real Rec-RIR API equivalence,
batch-four FP32 forward/backward/AdamW, frozen source preservation, exact zero
Glue positions, 629 input/501 output frames, CTF projection gradients, and the
longest full-length validation observation. Training starts only after a passed
probe and initializes seed 0 anew; probe weights are discarded. Complete model
checkpoints embed the frozen estimator and restore the same Glue configuration.

The controller propagates `--variant concat_glue` to all stages and continuation
jobs. Only scheduler-confirmed walltime failures with checkpoint epoch progress
may resume; other failures stop the chain. It requires the reporter's actual
stopping criterion before full-utterance valid/test inference and eight-worker
CPU scoring. Both splits use `valid.loss.best.pth` and WAV peak normalization,
matching the baseline and avoiding PCM16 clipping.

Output: `exp/enh_train_tflocoformer_s_nf8k_recrir_ctf_concat_glue_seed0`.
Job history and completion markers are under its own `.pipeline`; `RESULTS.md`
contains the final split averages. Source, configuration, data manifest and code
identity are pinned for the entire run.

At priority -5, per-job upper estimates are 3.84 points for `gpu_1=1` training
(24 h), 0.64 for inference (4 h), 0.144 for `cpu_8=1` scoring (6 h), and 0.002
for each `cpu_4=1` controller (10 min). Runtime for this estimated-CTF Glue run
is not measured. Retry limits are 12 training, 2 inference and 3 scoring jobs;
the full retry allowance including controllers is at most 47.828 points. These
are upper estimates, not confirmed charges or training-completion predictions.
