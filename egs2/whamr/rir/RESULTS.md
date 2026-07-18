# Rec-RIR 2-speaker PIT Results

Date: 2026-07-13

## Evaluation setup

- Test set: `dump_rir_clean/raw/tt_mix_clean_reverb_min_8k`
- Input waveform scp: `speech_mix_pit.scp`
- Reference RIR scp: `rir.scp`
- Output: two mic-0 RIRs per mixture
- Sample rate: 8000 Hz
- RIR length: 8192 samples
- PIT metric: `rmse_50ms`
- Scoring options: `align=peak`, `scale_mode=peak`
- Number of utterances: 3000
- Number of source-RIR pairs: 6000

## Final Rec-RIR 2-speaker PIT checkpoints

| Checkpoint | Decode directory | rmse_mean | rmse_50ms_mean | corr_mean | rt60_mae | rt60_rmse | rt60_pearson | drr_mae | drr_rmse | drr_pearson | c50_mae | c50_rmse | c50_pearson | perm0_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `valid.loss.best.pth` | `exp/rir_train_rec_rir_2spk_pit_clean_8192/inference/tt_mix_clean_reverb_min_8k_valid.loss.best_20260713_145025` | 0.010236 | 0.037194 | 0.819253 | 0.104174 | 0.140126 | 0.940253 | 2.177247 | 2.506834 | 0.982613 | 4.699635 | 4.967294 | 0.988859 | 0.531333 |
| `valid.loss.ave_5best.pth` | `exp/rir_train_rec_rir_2spk_pit_clean_8192/inference/tt_mix_clean_reverb_min_8k_valid.loss.ave_5best_20260713_145024` | 0.010310 | 0.037586 | 0.816774 | 0.113693 | 0.163609 | 0.605045 | 3.001699 | 3.432566 | 0.950802 | 4.894933 | 5.599324 | 0.931118 | 0.533667 |
| `36epoch.pth` | `exp/rir_train_rec_rir_2spk_pit_clean_8192/inference/tt_mix_clean_reverb_min_8k_36epoch` | 0.010313 | 0.037475 | 0.817302 | 0.125197 | 0.173891 | 0.769861 | 2.244238 | 2.611554 | 0.977774 | 4.816322 | 5.119771 | 0.985981 | 0.538667 |

Selected checkpoint: `valid.loss.best.pth`.

## Baseline comparison

| Condition | rmse_mean | rmse_50ms_mean | corr_mean | rt60_mae | rt60_rmse | rt60_pearson | drr_mae | drr_rmse | drr_pearson | c50_mae | c50_rmse | c50_pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rec-RIR 1spk 66epoch, dump `rir1.scp` reference | 0.010129 | 0.036895 | 0.822482 | 0.063578 | 0.094928 | 0.954571 | 2.030138 | 2.367801 | 0.984007 | 4.171671 | 4.485794 | 0.990926 |
| Rec-RIR 2spk PIT `valid.loss.best.pth` | 0.010236 | 0.037194 | 0.819253 | 0.104174 | 0.140126 | 0.940253 | 2.177247 | 2.506834 | 0.982613 | 4.699635 | 4.967294 | 0.988859 |
| TF-Locoformer 2spk PIT `valid.loss.best` | 0.010345 | 0.037553 | 0.815532 | 0.134263 | 0.178558 | 0.889807 | 2.401678 | 2.738722 | 0.977797 | 5.240298 | 5.551148 | 0.984776 |
| Rec-RIR 1spk estimate duplicated as 2spk PIT | 0.012647 | 0.048202 | 0.698489 | 0.063402 | 0.091863 | 0.963309 | 2.698232 | 3.443321 | 0.877535 | 4.300444 | 4.790537 | 0.975180 |
| Random different-sample RIR pair, 5-seed mean | 0.016628 | 0.064903 | 0.587636 | 0.201353 | 0.250360 | 0.000611 | 6.238404 | 8.355127 | 0.007364 | 10.657478 | 13.929120 | -0.000818 |

## Notes

- `run_infer_rec_rir_2spk_pit_clean_8192.sh` and `run_score_rec_rir_2spk_pit_clean_8192.sh` generate `summary.json`; they do not generate this `RESULTS.md` automatically.
- The 2-speaker PIT result is close to the 1-speaker Rec-RIR score under peak alignment and peak scaling, and clearly better than the random different-sample baseline.
- The duplicated 1-speaker estimate is worse than the trained 2-speaker PIT model on RMSE and correlation, so the 2-speaker model is not just reproducing one identical RIR twice under this scoring.
- The original 1-speaker score against `rir_ref.scp` has one RT60 reference outlier. The table above uses the dump `rir1.scp` reference for the 1-speaker comparison to avoid that file-format dependent RT60 artifact.
- Because scoring uses `align=peak` and `scale_mode=peak`, direct-path delay and absolute scale errors are mostly absorbed by scoring. For source-position-specific analysis, also inspect no-alignment scores and direct-peak delay error.
