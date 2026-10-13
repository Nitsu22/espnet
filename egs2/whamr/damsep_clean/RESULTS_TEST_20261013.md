# DAMSEP NF-WHAMR test evaluation (2026-10-13)

Both trained models were evaluated on all 3,000 NF-WHAMR 8 kHz/min test
mixtures, left channel, against the same clean references as `enh4_clean`.
Inference uses 0.9 peak normalization and PCM16 WAVs. Scores use the standard
ESPnet enhancement scorer, including PIT source matching. Metrics are absolute
values, not improvements over the mixture; STOI is expressed as a percentage.

| Model | Selected epoch | SI-SNR (dB) | SDR (dB) | SIR (dB) | SAR (dB) | STOI (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAMSEP, three losses | 29 | 13.50 | 14.76 | 26.70 | 15.11 | 96.05 |
| DAMSEP, clean waveform loss only | 10 | 11.19 | 12.71 | 23.92 | 13.13 | 94.18 |
| Existing TF-Locoformer S | — | 18.28 | 19.45 | 32.23 | 19.74 | 98.09 |
| Existing TF-Locoformer M | — | 20.65 | 21.78 | 34.90 | 22.03 | 98.79 |

The three-loss model exceeds clean-only by 2.31 dB SI-SNR and 2.05 dB SDR.
It remains 4.78 dB and 7.15 dB below S and M, respectively, in test SI-SNR.
DAMSEP used one GPU with effective batch one; the existing baselines used four
GPUs. Early stopping occurred before any learning-rate reduction in both
DAMSEP experiments. These runs do not establish the cause of the performance
gap or isolate auxiliary-loss effects from differences in the stopping epoch.

## Verification and provenance

- Saved validation-best models: three-loss `29epoch.pth`, clean-only
  `10epoch.pth`. Selection used each run's `valid.loss` criterion.
- GPU inference job `8993603`: 12:23:18–12:54:43 JST, exit status 0,
  scheduler failure 0, wall time 1,884.758 seconds.
- CPU scoring job `8993604`: 12:54:57–13:10:42 JST, exit status 0,
  scheduler failure 0, wall time 944.600 seconds.
- Both inference reports record 3,000 utterances, sample rate 8,000 and
  channel 0. Before scoring, the output SCPs were checked against the exact
  test ID sequence and all referenced waveform files were checked to exist.
- Each clean score file has exactly 3,000 unique utterance IDs per speaker.
  All 6,000 values per metric are finite; their means reproduce the table.
- Evaluation script/code commit: `57f0d4a57f03b6a11c083380e5a722307d4a4a81`.
  Training source commits: three-loss `72526c93eac`, clean-only `152e7dbe7b`.

TSUBAME experiment roots:

```text
/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/damsep_clean/exp/damsep_nf_8k
/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet_damsep_clean_only/egs2/whamr/damsep_clean/exp/damsep_nf_8k_clean_only
```

Detailed outputs are under each root's `enhanced_tt/clean/scoring/`.
Baseline results were read from `RESULTS.md` in the primary checkout under
`egs2/whamr/enh4_clean/exp/`:

- `enh_train_enh_tflocoformer_nocashe_clean_small_4gpu`
- `enh_train_enh_tflocoformer_nocashe_clean_medium_4gpu`

## Intermediate reverberant output

The three-loss model's intermediate reverberant output was also scored against
the reverberant references: SI-SNR **−6.61 dB**, SDR 2.46 dB, SIR 15.81 dB,
SAR 2.83 dB and STOI 75.06%. This weak intermediate result needs investigation.
It is separate from the clean-output comparison above. CTF estimates were
exported but physical RIR fidelity and distance ordering were not evaluated.

## Point estimates

Both jobs used priority −5. GPU inference requested `gpu_1=1`, two hours;
CPU scoring requested `cpu_8=1`, four hours. Their original upper estimates
were 0.3200 and 0.0960 points, total 0.4160.
Using the scheduler-reported wall times, the calculated estimates are
0.1132 and 0.0175 points (floored separately to four decimals), total 0.1307.
These are estimates, not confirmed charges.
