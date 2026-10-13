# NF-WHAMR clean architecture ablations

Both experiments use the same clean waveform PIT negative-SNR objective,
validation negative-SI-SDR, NF-WHAMR 8 kHz data, seed 0, four-second train
segments, full validation utterances, FP32, one GPU, batch one and Adam
learning rate 0.001 as the existing clean-only experiment. Its scheduler
and five-consecutive-failure early stopping are preserved. Each experiment
starts in an independent output directory and uses its own checkpoint.

| Variant | Forward path | Active parameters |
| --- | --- | ---: |
| `spmamba_clean` | STFT → SPMamba → two-source complex decoder → iSTFT | 6,094,498 |
| `feature_dereverb_clean` | STFT → SPMamba features → Linear(16,64) → two Dereverb blocks → joint two-source complex decoder → iSTFT | 6,378,370 |

`model_conf.architecture` selects `spmamba` or `feature_dereverb`.
Both require `clean_only: true` and zero auxiliary loss weights. The
default `damsep` architecture keeps the existing full and clean-only paths.

The standalone variant retains the SPMamba separator decoder. It skips the
entire Dereverb/CTF network. Common separator weights have identical seed-0
initialization across the three architectures. Inactive modules are frozen.

The feature variant replaces the intermediate source-spectrum bottleneck
with a pointwise linear expansion from 16 to 64 features. It skips the
separator ConvTranspose2d, intermediate iSTFT/STFT and Dereverb Conv2d
encoder. This expansion does not create independent information beyond
its 16-dimensional input. The Dereverb blocks retain their original hidden
width 64 and internal cross-band processing.

Direct mixture features use the separator's 256-point FFT, 64-sample hop
and 129 frequency bins. This changes the previous Dereverb's 512-point FFT,
128-sample hop and 257 bins. Since source-specific waveforms have not yet
been decoded, the final decoder jointly produces four real/imaginary
channels for two sources, instead of applying a one-source decoder to
each speaker independently. Input standard deviation is restored at the
final waveform output. These changes accompany the removal of the
intermediate decoder/encoder and should be included when interpreting
the comparison.

## Data and execution

Use an independent TSUBAME checkout at the pushed work-branch commit.
Link existing `enh4_clean/dump_clean`, `enh_rir/dump_ctf_joint`,
`damsep_clean/.deps` and `exp/damsep_stats_8k`; Stage 5 needs no rerun.
Submit from `egs2/whamr/damsep_clean`:

```bash
python local/ablation_pipeline.py submit --variant spmamba_clean
python local/ablation_pipeline.py submit --variant feature_dereverb_clean
```

Each pipeline starts a GPU job and a CPU controller waiting for that job.
The first allocation verifies Stage-5 source lists, the longest real
validation utterance, a real training backward/Adam update and the longest
test utterance's CUDA inference path. GPU jobs run Stage 6 and Stage 7.
Time limits trigger checkpoint continuation through the controller;
other failures stop the pipeline. Once inference is complete, Stage 8
runs in a separate CPU allocation. Completion requires all test keys and
finite per-speaker scoring results. Job IDs and state are recorded under
each experiment's `pipeline/` directory.

| Variant | Config | Output |
| --- | --- | --- |
| `spmamba_clean` | `conf/tuning/train_damsep_nf_8k_spmamba_clean.yaml` | `exp/damsep_nf_8k_spmamba_clean` |
| `feature_dereverb_clean` | `conf/tuning/train_damsep_nf_8k_feature_dereverb_clean.yaml` | `exp/damsep_nf_8k_feature_dereverb_clean` |

GPU chunks request `gpu_1=1`, 24 hours and priority -5, giving an upper
point estimate of `0.2 * 0.8 * 24 = 3.8400` per chunk. Controllers request
`cpu_4=1` for 20 minutes (upper 0.0040); CPU scoring requests `cpu_8=1`
for two hours (upper 0.0480). Convergence time and the total number of
GPU chunks are unknown. These are estimates, not confirmed charges.

Stage 7 selects `valid.loss.best.pth`, exports only clean WAVs and uses
the baseline's 0.9 peak normalization and PCM16 encoding. Stage 8 uses
the same clean references, left channel and ESPnet PIT scoring. SI-SNR,
SDR, SIR, SAR and STOI are absolute metrics, not improvements over the
mixture. Individual results are under
`enhanced_tt/clean/scoring/result_*.txt`.

## Validation

The model tests cover forbidden decoder/encoder calls, batch-two odd-length
outputs, finite and nonzero active gradients, silent/quiet waveform scaling,
PIT, common seed-0 backbone weights, native save/resume and clean-only
inference/scoring. Pipeline tests exercise resource requests, timeout
continuation, failure gating and score-completion checks. CUDA checks run
inside the charged allocation before production training.
