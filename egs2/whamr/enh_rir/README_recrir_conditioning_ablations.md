# Frozen Rec-RIR conditioning on NF-WHAMR 8 kHz

Compare three independent changes against the current MHCA + FiLM separator.
All runs use seed 0, batch 4, FP32, four-second aligned crops, four TF-Locoformer
blocks with D=96, maximum 150 epochs and the existing early-stopping/LR rules.
Freeze the original native-8-kHz NF Rec-RIR epoch-97 checkpoint, SHA256
`5fe27c7996e70d290d8646fb39f9cb6fe6a3f2972138e5f16b42730c8f4cafc8`.
The checkpoint is the same as the ongoing Rec-RIR temporal-concat experiment.

| Pipeline variant | Only change relative to reference |
| --- | --- |
| `film` | Reference frequency-wise MHCA + linear gamma/beta heads |
| `lag_position` | Add fixed sinusoidal lag positions to normalized CTF features |
| `joint_mlp` | Per layer, gamma/beta heads receive SiLU(Conv1x1([speech; context])) |
| `multiscale` | First 48 CTF output channels use lag dilation 1; last 48 use 8 |

For lag position l and feature pair i, p[l,2i] = sin(l/10000^(2i/D)) and
p[l,2i+1] = cos(l/10000^(2i/D)). Lag l represents l*16 ms. This is an absolute
CTF delay coordinate, not the separator's 8-ms audio-frame coordinate.

Multiscale uses the same 96x4x3x3 convolution parameters and shared GroupNorm
as the reference. Its lag receptive fields are 3 and 17 taps, respectively;
frequency dilation remains 1. There is no mean pooling or extra parameter.
All existing parameter names, construction RNG and common Xavier draws are
preserved. Joint fusion modules are appended after existing separator parameters
and constructed under fork_rng. FiLM starts at gamma=1 and beta=0 in every run.
The joint variant adds 74,112 trainable parameters (18,528 per block).

Prepared NF inputs (tr=20,000, cv=5,000, tt=3,000) and the frozen checkpoint are
verified by hashes. CTF is estimated from the actual input crop without reference
speech. Stage 7 uses valid.loss.best.pth and output normalization for every run.
Stage 8 scores the complete cv and tt sets on CPU. Select experimental designs
using validation, not the test metrics. These single-seed runs are screening;
they do not establish seed-robust gains or isolate additional MLP capacity.

Submit the controller from this recipe in a dedicated, pinned checkout:

```sh
qsub -g tga-shinoda -p -5 -l cpu_4=1 -l h_rt=00:10:00 \
  qsub/nf8k_recrir_pipeline.sh --variant lag_position --phase control
```

Replace lag_position with film, joint_mlp or multiscale. The controller chains
training, inference and scoring, checking exit status and checkpoint progress.
Each GPU allocation first checks actual batch-four training, frozen weights,
native CTF conventions, gradients and longest validation input. On a genuine
timeout, resume only with checkpoint progress; other failures stop the chain.

Resource budget per run, priority -5: at most nine gpu_1 training segments of
24 h (34.56 points), one gpu_1 inference segment of 4 h (0.64), one cpu_8 scoring
segment of 6 h (0.144), and at most twelve cpu_4 controllers of 10 min (0.024).
The scheduler-request upper total is 35.368 points per run, 141.472 for four.
This is a spending bound, not an estimated bill or a guarantee of convergence
within that time. Reaching the limit leaves the checkpoint and stops submission.
The earlier ~18-point estimate was measured with the faster BiMamba predictor;
use actual Rec-RIR epoch durations for the revised runtime forecast.
