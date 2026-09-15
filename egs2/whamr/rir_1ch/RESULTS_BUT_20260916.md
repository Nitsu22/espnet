# BUT ReverbDB evaluation — 2026-09-16

All three approved models completed clean/noisy inference and scoring on
3000 utterances per condition (18000 total), exit status 0. Every reported
metric is valid for all 3000 utterances in each model/condition.

Models: scratch-trained NF-WHAMR, 16 kHz, input-only folded batching.
Validation-loss-best checkpoints were copied before inference and hashed:

- Rec-RIR (three losses): epoch 17, aurum GPU 0.
- TF-Locoformer CTF: epoch 43, shannon GPU 4.
- TF-Locoformer original sweep loss: epoch 44, shannon GPU 6.

These are snapshots, not an equal-epoch or parameter-matched architecture
ablation. The additional RIR-L1 sweep variants were not included.

## Results

| Model | Condition | RT60 MAE (s) | DRR MAE (dB) | C50 MAE (dB) | RIR-50ms RMSE | Correlation |
|---|---|---:|---:|---:|---:|---:|
| rec_rir | clean | 0.574865 | 2.647074 | 1.795811 | 0.157289 | -0.140285 |
| rec_rir | noisy | 0.567379 | 5.314749 | 2.997414 | 0.153134 | -0.139268 |
| tflocoformer_ctf | clean | 0.587433 | 2.428664 | 1.998052 | 0.158860 | -0.147240 |
| tflocoformer_ctf | noisy | 0.588247 | 4.377059 | 3.304135 | 0.155358 | -0.145766 |
| tflocoformer_sweep | clean | 0.579741 | 2.591612 | 2.361420 | 0.160457 | -0.133902 |
| tflocoformer_sweep | noisy | 0.581951 | 4.263294 | 3.505115 | 0.156167 | -0.135094 |

RT60 and C50 MAEs are lowest for Rec-RIR in both conditions. DRR MAE is
lowest for TF-Locoformer CTF on clean inputs and TF-Locoformer sweep on noisy
inputs. No statistical-significance claim is made.

## Protocol and limitations

Use dump_but's same 3000 WHAMR tt/min speaker-1 utterances, microphone 01,
9 rooms and 51 RIRs. Clean inputs are reverberant speech; noisy inputs add WHAM
noise using the stored gains, not a fixed SNR. This is held-out evaluation.

Inference uses full utterances in FP32, outputs 32000-sample FLOAT WAVs,
and peak-normalizes predictions. Scoring uses the existing ACE protocol:
absolute-peak alignment, 40 pre-peak samples, independent peak normalization
preserving polarity, 2-second windows, RT60 from Schroeder T30/T20 regression,
DRR with a +/-2.5 ms direct window, and C50 after the peak. Waveform metrics
cover peak through +50 ms. This is not an official SimACE score or a comparison
against publisher-provided acoustic annotations. The scorer's legacy JSON
scope label says WHAMR-ACE; the actual dataset here is BUT, as recorded in
provenance.json and the input SCPs.

BUT's selected distributed RIRs are only 1 second long; unavailable reference
samples are zero-padded. The model has 60 CTF taps at a 256-sample hop (about
0.96 seconds), independent of its 2-second saved output. Room-level results
show substantial RT60 underestimation in long-reverberation rooms; these scores
alone do not identify its cause.

Signed waveform correlation is affected by polarity: 28 of 51 reference RIRs
have negative absolute peaks, covering 1915 of the 3000 utterances. In the
clean TF-Locoformer results, mean correlation is negative for that group and
positive for the positive-peak group. Therefore raw signed correlation is not
interpreted as shape accuracy alone. For transparency the original signed
metrics above are unchanged; the supplementary metric below is mean(abs(Pearson))
over utterances, allowing a global sign flip per estimate. It does not alter
RT60/DRR/C50 or select/retrain any model.

| Model | Condition | Mean absolute waveform correlation |
|---|---|---:|
| rec_rir | clean | 0.378687 |
| rec_rir | noisy | 0.376891 |
| tflocoformer_ctf | clean | 0.379611 |
| tflocoformer_ctf | noisy | 0.375944 |
| tflocoformer_sweep | clean | 0.355442 |
| tflocoformer_sweep | noisy | 0.355543 |

## Artifacts and reproduction

Results root: exp/but_comparison_20260916/.

- provenance.json: source experiments, exact epoch files, model/config SHA256,
  inference code commit and dataset generation-config hash.
- MODEL/snapshot/: frozen config and checkpoint copies.
- MODEL/{clean,noisy}/rir/: predicted RIRs and wav.scp.
- MODEL/{clean,noisy}/score/: all utterance scores and per-RIR summaries.
- comparison.json: overall and per-room summaries, supplementary correlation.
- comparison.md: combined primary metrics.
- MODEL/job.sh, run.log, exit_status: executed jobs and completion status.
- summarize.py: regenerate combined summaries from completed score files.

The evaluation runner accepts --dataset but_clean or but_noisy and --rir_exp
pointing to the frozen snapshot directory. Use a new --output_dir when rerunning.
