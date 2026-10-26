# Same-speaker mixture stress test (NF-WHAMR 8 kHz, Small)

This is an **evaluation-only** comparison of the completed Baseline S and
frozen-BiMamba estimated-CTF / cross-attention + FiLM ALL S. No training data,
validation data, model architecture, checkpoint selection, or optimizer changes.
It is not the standard NF-WHAMR test benchmark.

## Definition

For each of the original 3,000 test scenes:

- Retain the room, microphone, two source positions and their physical RIRs.
- Choose one of the original speakers as an anchor. Retain that speaker's
  original utterance at its original source position; replace the other source
  with a **different utterance of the same person**. Thus both sources now
  have the same speaker identity, not the same waveform.
- Choose anchors with seed 0, balanced between source positions (1,500 each).
  A candidate must be long enough to preserve the exact original mixture
  duration. Feasibility is determined before any model output is inspected.
- Use only the 1,770 unique utterances in the original test CSV. Their utterance
  IDs AND speaker IDs must be disjoint from the train/validation CSVs.
- Convolve raw 16-kHz speech with the saved **16-kHz** physical RIR, then
  downsample to 8 kHz and truncate to the original duration. Convolving already
  downsampled speech with the 8-kHz RIR is not equivalent and is not used.
- First match the original prepared anechoic reference's mean-square power at
  each source position. Both the direct and reverberant versions of that source
  use the same gain. This preserves the reference power ratio, not a claim of
  identical reverberant-source power or acoustic SIR.
- If the new mixture or a reference exceeds the PCM16 range, apply ONE common
  attenuation to the mixture and both references (peak 0.99). Do not clip or
  reselect the donor. This preserves the reference power ratio; only the common
  absolute scale changes. Both fixed models use `normalize_variance: true`
  before inference, so this common scale is removed apart from PCM16 rounding.
  The per-scene common gain and attenuation counts are recorded in the audit.
- Write channel 0 as mono PCM16, matching the original sample rate,
  quantization and anechoic-reference definition. Refuse clipping, silence,
  zero-padding, tiling, key mismatches or more than 0.01-dB reference-power error.
- Before accepting each new scene, reconstruct the original two-speaker mix
  and references from raw speech, RIRs and original **8-kHz** generation gains.
  Maximum absolute error against the prepared PCM16 files must be <= 5e-5.

The original scene keys are retained for paired comparisons and room-metadata
joins. **The speaker prefixes in those keys describe the OLD speakers**;
`audit.tsv` and `utt2spk` record the actual new speaker identities. The audit
also records source/RIR hashes, anchor positions, gains, power and control errors.

BiMamba estimates a new CTF on the actual new mixture during inference. No
original estimated CTF, Oracle CTF, clean-reference matching, or speaker-order
constraint is supplied to the model. Standard two-source PIT scoring remains
appropriate even though the two references have the same speaker identity.

## Prepare only test (from enh_rir, on midgar)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  local/prepare_nf8k_same_speaker_test.py \
  --baseline-test exp/bimamba_nf8k_baseline_inputs/raw/tt_mix_clean_reverb_min_8k \
  --baseline-recipe ../enh7_baseline \
  --raw-root data/wsj0/wsj0_wav --physical-root data_rir_plus \
  --csv-dir whamr_scripts/data --output dump_nf8k_same_speaker_seed0 \
  --destination-root /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet_nf8k_same_speaker_test/egs2/whamr/enh_rir/dump_nf8k_same_speaker_seed0 \
  --seed 0 --workers 4 --peak-policy common-attenuate
```

The destination prefix is optional; use it when the generated dump will be
transferred to TSUBAME. Audio is written locally, SCPs reference that final
absolute destination. Transfer the entire new dump, then verify ALL files
against `preparation.json:output_hashes` on the destination before inference.
An existing output dump is never overwritten.

## Fixed checkpoints and separate evaluation output

Baseline experiment:

`/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh7_baseline/exp/enh_train_tflocoformer_s_nf_8k_1gpu_batch4`

BiMamba FiLM experiment:

`/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet_nf8k_bimamba_conditioning/egs2/whamr/enh_rir/exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0`

`local/evaluate_same_speaker_test.sh` accepts `--test_dir`, `--baseline_exp`,
`--conditioned_exp`, `--output_dir`, `--stage 7`, `--stop_stage 8`.
It resolves each `valid.loss.best.pth` to an immutable epoch file and records
config/checkpoint hashes before inference. New output is required; explicit
`--resume true` verifies the original pins and only skips complete outputs.
Neither training experiment is written to. The native training wrapper is not
used, because it also processes CV.

For TSUBAME, sync this code into an isolated, inactive Git checkout and submit
from its `enh_rir` directory. Set `TEST`, `BASE`, `FILM`, `OUT`, `ORIG_BASE_SCORE`,
`ORIG_FILM_SCORE` to the explicit paths above and desired separate output.

```bash
qsub -g tga-shinoda -l gpu_1=1 -l h_rt=01:00:00 \
  qsub/eval_same_speaker_test.sh infer "$TEST" "$BASE" "$FILM" "$OUT"
qsub -g tga-shinoda -l cpu_4=1 -l h_rt=01:30:00 -hold_jid INFERENCE_JOB_ID \
  qsub/eval_same_speaker_test.sh score "$TEST" "$BASE" "$FILM" "$OUT" true \
  "$ORIG_BASE_SCORE" "$ORIG_FILM_SCORE"
```

Both models infer sequentially on one GPU. CPU scoring checks complete
inference rather than assuming that a hold dependency succeeded. Models use
the same full-waveform inference, batch 1, 8 kHz, reference channel 0, and
ESPnet BSS_EVAL permutation as the completed benchmark.

Output WAV peak normalization is enabled by default for BOTH models, matching
the Baseline benchmark and preventing PCM16 clipping. The setting is pinned
in `provenance.json`; resuming an old unnormalized run requires explicit
`--normalize_output_wav false` on the local runner. Use a new evaluation output
for the normalized Baseline / temporal-concat comparison; do not reuse old scores.

## Results and interpretation

`OUT/summary.csv` holds all six standard ESPnet metrics. The optional original
score-directory arguments also create `OUT/analysis_job_JOB_ID/RESULTS.md`,
`summary.csv`, `paired_scenes.csv`, and provenance. These compare original vs
same-speaker model gains, overall and in predeclared RT60 groups
[0.1,0.3), [0.3,0.6), [0.6,1.0]. Bootstrap uses the **paired scene** as its unit
(both sources and both conditions together), 5,000 draws, seed 0.

This probes whether the conditioning helps when voice identity cannot
distinguish the two signals. It does not establish that the predictor encodes
only room acoustics: utterance content also changes, speech recurs across
scenes, the data are outside the standard two-speaker test distribution, and
only one trained seed is being evaluated. A larger gain would be evidence for
this stress-test condition, not by itself a causal proof.
