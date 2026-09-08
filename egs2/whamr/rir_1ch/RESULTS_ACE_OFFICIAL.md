# Released Rec-RIR checkpoint on WHAMR-ACE

Completed 2026-09-09 using ESPnet implementation commit
`3ea153d2c88e858ac310cc38bb95df1861576ee9` on aurum (RTX 3090), with
the existing `recrir` environment, PyTorch 2.7.1+cu118. No retraining or
environment modifications were performed.

## Checkpoint and implementation verification

- Official checkout: `/net/midgar/work2/nitsu/learning/Rec-RIR`
- Official commit: `6914eea792f0117e39f87510ffbda7cd04dffbbb`
- Official inference model/acoustics/PIM/config/checkpoint had no tracked changes.
- Configuration: `config/Rec-RIR.toml`; checkpoint: `ckpt/epoch35.tar`.
- Checkpoint SHA256: `afe36e477f6fb3656d162d906a47da2d40a789634b476564266fc1ba36037478`.
- Strict weight loading succeeded for the ESPnet and official networks.
- One full utterance per ACE RIR in each noise condition: 28 parity examples.
- Maximum absolute CTF difference: **0**. Maximum absolute normalized RIR
  difference: **0**, across all 28 examples on this GPU/environment.
- Parity compared in-memory outputs before audio serialization, with
  `atol=1e-5, rtol=1e-4`. It gates full inference in the launcher.

## Evaluation results

Each condition has 3000 utterances and the same 14 measured ACE Single RIRs.
This evaluates the released model on the generated WHAMR-ACE test set; it does
not reproduce SimACE or isolate architecture effects from training-data effects.

| Metric | Clean | Noisy |
|---|---:|---:|
| RT60 Schroeder MAE (s) | 0.201632 | 0.220919 |
| DRR MAE (dB) | 0.726091 | 1.463816 |
| C50 MAE (dB) | 0.586843 | 0.966671 |
| Peak-to-50-ms waveform RMSE | 0.037617 | 0.039571 |
| Peak-to-50-ms waveform correlation | 0.800994 | 0.778964 |

These are utterance-weighted means. All five metrics have 3000 valid values and
zero invalid values in each condition. Per-RIR results are retained in the
condition summaries; 3000 utterances do not constitute 3000 independent rooms.

RIRs are independently absolute-peak normalized with polarity preserved,
aligned **before** cropping, and evaluated over 32000 samples at 16 kHz with
40 samples before the peak. The maximum reference energy outside this window
is 0.0261072%. RT60 is computed by Schroeder regression on the -5 to -35 dB
decay (fallback -5 to -25 dB), extrapolated to -60 dB; it is not the ACE
annotated RT60. DRR uses +/-2.5 ms around the peak. C50 compares energy from
the peak to +50 ms with the remaining tail. Both prediction and reference
use the same two-second support.

The noisy condition preserves WHAMR noise/speech gains. Full-reverberant-speech
SNR ranges from -9.439 to 13.325 dB, median 3.041 dB. This is not a fixed-SNR
test. The 8 kHz data was not evaluated with this 16 kHz checkpoint.

## Saved artifacts and reproduction

Run `bash local/evaluate_official_rec_rir_ace.sh` on an available lab GPU in
`tmux`, following the repository GPU skill. Use a new `OUTPUT_DIR` for reruns.
Existing output directories are refused. The entry point is
`espnet2.bin.rec_rir_official_inference`; it reuses ESPnet's network, transforms
and PIM while accepting the released checkpoint and official TOML directly.

All run artifacts are under `exp/official_rec_rir_ace_16k/`:

- `clean/`, `noisy/`: 3000 FLOAT RIR WAVs each, `wav.scp`, provenance,
  completion markers, and full waveform validation reports.
- `parity_clean/parity.json`, `parity_noisy/parity.json`: all parity errors
  and hashes of the official source files used in the comparison.
- `score_clean/summary.json`, `score_noisy/summary.json`: full-precision
  aggregate and per-RIR scores and evaluation definitions.
- `score_clean/utterances.jsonl`, `score_noisy/utterances.jsonl`: per-utterance scores.
- `reference_window_audit.json`, `noise_condition.json`, `run.log`.

All 6000 saved WAVs were checked for 16 kHz, exactly 32000 samples, finite
values, unit absolute peak, and peak at sample 40. Synthetic delayed exponential
and two-tap RIR checks verified alignment, a known 0.6-second RT60, analytic
DRR, undefined C50 handling, and rejection of invalid input. The GPU job exited
and its tmux session ended automatically after completion.
