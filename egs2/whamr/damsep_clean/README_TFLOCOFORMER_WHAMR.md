# TF-Locoformer S + DAMSEP on noisy WHAMR

These two fresh experiments replace the SPMamba separation stage with the
unmodified TF-Locoformer-S from `../enh7_baseline`, keeping DAMSEP's shared
speaker dereverberation stage and optional CTF branch.

| Variant | Clean SI-SNR | Reverb RI+Mag | GT-clean-to-reverb CTF reconstruction |
|---|---:|---:|---:|
| `clean_reverb` | 1 | 0.1 | 0 (CTF execution/update disabled) |
| `three_loss` | 1 | 0.1 | 0.5 |

The input is noise + two reverberant speech sources. Clean references are the
same anechoic teachers as the noisy WHAMR baseline. Each auxiliary target is
one speaker's noise-free reverberant speech, never the noisy mixture. A single
PIT permutation selected from the final clean SI-SNR applies to all outputs.
Reconstruction filters **reference clean** spectra with predicted CTFs.

The TF branch uses FFT256 / hop64 Hann, per-mixture variance normalization,
four blocks / embedding96 and baseline Xavier initialization. Its complex
outputs are decoded and restored to the original waveform scale before the
shared dereverberation/CTF stage (FFT512 / window512 / hop128 Hann). The latter
keeps the released hidden64, two dereverb blocks, six CTF narrowband blocks,
60 complex CTF taps and scalar fusion. Auxiliary losses retain DAMSEP's
separate FFT512 / window256 / hop128 sqrt-Hann transform. The CTF branch is
constructed in both experiments to preserve identical initialization of all
shared parameters; its parameters are frozen and its execution skipped in A.
Mamba remains necessary for the shared stage, even though TF replaces SPMamba.

Baseline training settings are retained: seed0, FP32, 8 kHz left channel,
at most four-second aligned crops, short utterances retained, physical batch4,
accumulation1, validation batch1, AdamW lr0.001 / weight decay0.01, gradient
clip5, 4000-update warmup, plateau factor0.5 / patience3, max150 epochs,
ESPnet early-stop patience10 (11 consecutive non-improving epochs). Both
conditions monitor **valid.loss_clean** for scheduling, early stopping and
single-best-model selection. This is a TF baseline extension, with SI-SNR and
AdamW replacing the DAMSEP paper's training SNR and Adam.

Existing artifacts are linked, never regenerated:

- `dump_whamr` -> existing `../enh1/dump` noisy mixture and clean teachers.
- `dump` -> the same noisy dump, to resolve its SCPs' original relative WAV paths.
- `dump_clean` -> existing `../enh4_clean/dump_clean`, for source checks only.
- `dump_reverb` -> existing `../enh_rir/dump_ctf_joint`, speech-only reverb teachers.
- `exp/tflocoformer_damsep_stats_8k` -> the existing five-series DAMSEP statistics.
- `.deps` -> the already installed Mamba CUDA11.8/Torch2.1/Python3.10 overlay.

The checker validates actual noisy inputs against reused shape IDs/lengths,
all source sets/counts, sampled audio, cross-condition clean references and
`noisy = reverb1 + reverb2 + noise` for train/valid. The existing test dump has
no isolated noise SCP; inference/scoring only require mixture and references.
GPU preflight runs real physical-batch4 FP32 backward/AdamW and the longest
full validation/inference, records finite gradients and GPU memory, and stores
an initial-state SHA256 for the A/B comparison.

From `egs2/whamr/damsep_clean` in an allocation:

```bash
bash run_tflocoformer_damsep.sh --variant clean_reverb --stage 6 --stop_stage 6
bash run_tflocoformer_damsep.sh --variant three_loss --stage 6 --stop_stage 6
```

Independent outputs are `exp/tflocoformer_damsep_whamr_8k_<variant>/`.
Repeating the same variant resumes its own checkpoint. Use an isolated TSUBAME
checkout; never update a directory referenced by queued/running jobs.

The production pipeline runs Stage6, resumes only scheduler-confirmed 24-hour
terminations with checkpoint progress, then Stage7 using
`valid.loss_clean.best.pth`, and Stage8 in a separate CPU allocation:

```bash
qsub -g tga-shinoda qsub/tflocoformer_damsep_data.sh
# After the CPU check succeeds:
python3 local/tflocoformer_damsep_pipeline.py submit --variant clean_reverb
python3 local/tflocoformer_damsep_pipeline.py submit --variant three_loss
```

Every actual job ID/config hash/commit/phase/result is stored under
`exp/.../pipeline/`. No automatic retries for numerical errors, OOM or unrelated
failures. Each GPU chunk is gpu_1=1 / 24h / priority−5 (upper estimate3.8400
points); each controller cpu_4=1 / 20min (upper0.0040); final scoring
cpu_8=1 / 2h (upper0.0480). First GPU+controller for both: upper7.6880 points,
then CPU scoring upper0.0960 total. Continuations are additional allocations;
convergence time and confirmed charges are not yet known. The shared initial
CPU data check is cpu_4=1 / 10min (upper0.0020 additional points).

Clean test scoring uses the unchanged ESPnet enhancement scorer and the same
PCM16/0.9 peak normalization as the baseline. The existing noisy WHAMR S
baseline's test SI-SNR11.46 / SDR12.47 dB is an absolute score, not improvement.
Reverberant outputs are also scored. CTF fidelity/distance metrics are separate
from this experiment's clean-separation comparison.
