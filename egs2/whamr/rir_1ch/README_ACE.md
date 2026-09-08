# WHAMR! tt/min, one speaker, ACE measured RIR evaluation

Generate from the ESPnet checkout with its numpy/scipy/soundfile environment:

```bash
python egs2/whamr/rir_1ch/local/create_whamr_ace_dump.py
```

Default output: `egs2/whamr/rir_1ch/dump_ace/`. No intermediate `data/`.
The output must not already exist. `--limit 3 --output /tmp/ace-smoke` creates
an independent smoke test, with the same assignments as the full dataset.
`--validate-only` rechecks the completed output without generating it again.

## Definition

- 3,000 WHAMR tt mixture entries; use speaker 1 only. Speaker 2 determines the
  original `min` duration but is never mixed into the output.
- 14 measured ACE Single RIRs, assigned evenly using a fixed seed. Same
  assignment at 8/16 kHz and for clean/noisy conditions.
- WSJ0 source files are read before acoustic simulation. Follow original
  wsjmix scaling, int16 quantization, WHAM speech scaling, min cropping, and
  WHAM noise scaling/crop start. No pyroomacoustics or artificial RIR generation.
- Resample each complete ACE RIR with scipy resample_poly and normalize its
  absolute peak to one. Keep the acquisition delay and full tail. Store the
  float32 RIR actually used in the convolution, plus the original 48 kHz file.
- Direct reference: retain the single signed peak tap of that RIR and convolve
  it with the source (VINP SimACE convention). This is an operational reference,
  not a claim that a measured RIR's largest peak is always its physical onset.
- One common attenuation factor per utterance/rate is applied to source,
  direct reference, reverberant speech, noise and noisy speech; it is also
  shared by clean/noisy conditions. Signal identities remain consistent.
- Keep WHAM noise gains, not a new fixed 20 dB SNR. Log both direct-reference
  and full-reverberant SNR. This is a WHAMR-derived test, not SimACE.
- Convolve before cropping to min length; do not reset filter state in later
  speech chunks. Full RIR tails are stored although audio is min-cropped.
- Do not use this test set for training, model selection or tuning RIR handling.

## Outputs and existing ESPnet readers

`raw/tt_ace_single_{clean,noisy}_reverb_min_{8k,16k}/` contains absolute paths:

- `wav.scp`, `speech_mix.scp`: the actual model input for this condition.
- `speech_reverb.scp`: noise-free reverberant reference in BOTH conditions.
- `speech_direct.scp`, `spk1.scp`: direct reference.
- `source.scp`: scaled source before convolution, not the direct reference.
- `rir_ref.scp`, `rir.scp`: complete convolution RIR.
- `utt2spk`, `spk2utt`, `utt2num_samples`, `feats_type`.

Audio and shared references reside under `audio/{8000,16000}/`; metadata is
in `generation_config.json`, `utterances.jsonl`, `rir_metadata.json`.
`generation_complete.json` is written only after generation completes;
`validation.json` only after full signal and SCP validation passes.

The existing `espnet2.bin.rec_rir_inference` accepts `--wav_scp` from these
sets and the matching `--sample_rate`. Provide an existing compatible model's
`--train_config` and `--model_file`. **Use wav.scp for noisy input**, not
speech_reverb.scp (some older wrappers hard-code the latter).
This is an evaluation-only dump; do not run rir.sh stages 1–4 to regenerate
it from a nonexistent data directory. There are no tr/cv sets here.

## Official Rec-RIR pretrained comparison

Run official `inference.py` with its own `config/Rec-RIR.toml` and
`ckpt/epoch35.tar`, using `-i dump_ace/audio/16000/reverb` or
`-i dump_ace/audio/16000/mix`. Do not pass the root audio directory (it also
contains references). Do not feed 8 kHz data to a checkpoint configured for
16 kHz or change its architecture configuration to 8 kHz.

The dataset stores full RIRs, not an evaluation-specific 8192-sample crop.
Official PIM returns a peak-relative waveform starting 2.5 ms before its
peak, up to two seconds, with peak normalization on save. A common evaluation
must specify alignment, sign/gain convention, frequency band, early-window
position and parameter estimator. The old local score_rir.py defaults (8192
samples, peak alignment/scale) are not a reproduction of official metrics.
Do not interpret increasing the saved length as increasing effective CTF
support. Comparing official pretrained weights is a system-generalization
comparison, not an architecture-only or published SimACE score reproduction.

Inspected official commits:
- Rec-RIR: b3ac6fc1421bd58e017022037feb14f30366b46f
- VINP: 8acbb6bc7eb98e4610612e0350b6117a4b7e95a5

ACE source: https://zenodo.org/records/6257551 (CC BY-ND 4.0).
Cite Eaton et al., "Estimation of Room Acoustic Parameters: The ACE Challenge",
IEEE/ACM TASLP, 2016. Original RIRs and generated data are kept out of Git.

## Released Rec-RIR checkpoint evaluation

`local/evaluate_official_rec_rir_ace.sh` uses the `recrir` environment on an
available lab GPU. It first compares ESPnet's existing network, transforms and
PIM with the official implementation on one utterance per ACE RIR, in both
conditions (28 checks). Both CTF and peak-normalized RIR must satisfy
`atol=1e-5, rtol=1e-4`; any failure stops the run before full evaluation.
The checkpoint is loaded strictly after removing the official `module.` prefix.
No training is performed. The official TOML supplies all network/acoustic settings.

Then all 3000 clean and 3000 noisy 16 kHz inputs are inferred. Outputs are FLOAT
WAVs with a `wav.scp`, provenance hashes, and a completion marker under
`exp/official_rec_rir_ace_16k/` (override with `OUTPUT_DIR`). Existing inference
output directories are rejected to prevent accidental mixing of runs.

`local/score_ace_rir.py` aligns each RIR by its absolute peak **before** cropping
or padding to 32000 samples, with 40 samples preceding the peak. It independently
peak-normalizes prediction and reference while retaining polarity. It reports
waveform RMSE/correlation from the peak to +50 ms, DRR with a +/-2.5 ms window,
C50, and an explicitly defined Schroeder RT60 estimate. Reference and predicted
tails share the same two-second window. These are WHAMR-ACE scores, not claimed
to reproduce official SimACE metrics or ACE annotated RT60 values. Undefined
metrics are recorded as null with valid/invalid counts. Per-RIR summaries are
included because there are only 14 distinct measured RIRs.
