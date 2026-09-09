# WHAMR audio and RIR generation

For Stage 1 only, run `bash run_rir_plus_stage1.sh --stage 1 --stop_stage 1`.
It defaults to stereo, both sample rates, both length conventions and all splits.
`--estimate_only true` scans input WAV headers without generating audio.
The generator checks that its header-based size estimate (including a 10% margin)
plus a 100 GiB reserve fits before starting. It rechecks free space before each
utterance and stops with an error if the next utterance would enter that reserve.
The reserve is configurable with `--min_free_gib` on the Stage 1 launcher.
This is not a filesystem reservation against other users' concurrent writes.
`capacity.json` records the estimate; `progress.json` updates every 500 utterances.

Run `bash run_generate_rir_plus.sh` to generate all splits (`tr cv tt`), both
sample rates (`8k 16k`), and both length conventions (`min max`). This is a
standalone audio-generation entry point, not the existing `run.sh` Stage 1.
It does not need to re-run WSJ transcription preparation or edit Python constants.

The default output is `enh_rir/data_rir_plus/`:

```text
data_rir_plus/
  generation_config.json
  generation_complete.json       # exists only after successful completion
  wav{8k,16k}/{min,max}/{tr,cv,tt}/
    mix_clean_anechoic/          # s1 + s2
    mix_clean_reverb/
    mix_single_anechoic/         # s1 + noise
    mix_single_reverb/
    mix_both_anechoic/           # s1 + s2 + noise
    mix_both_reverb/
    s1_anechoic/                 # spatial direct-path speech, not original WSJ
    s1_reverb/                   # one speaker without additive noise
    s2_anechoic/
    s2_reverb/
    noise/
    rir1_anechoic/
    rir1_reverb/
    rir2_anechoic/
    rir2_reverb/
    utterances.jsonl             # source paths, spatial gains, noise gain, timing
  manifests/{tr,cv,tt}_{min,max}_{8k,16k}/
    <signal-type>.scp            # one SCP for each of the 15 types above
    counts.json
```

All audio and RIRs are FLOAT WAV. Default channels are left/right; `--mono`
selects the left microphone for both audio and RIR. A room is simulated once
per utterance and reused for all sample rates and length conventions. Full RIR
length and propagation delay are retained. RIRs do not inherit speech cropping,
noise padding, utterance spatial gain, or peak normalization. Min/max copies
therefore share the same physical RIR at each sample rate.

RIR generation follows the existing `WhamRoom`: anechoic uses reflection order
zero and reverberant uses the configured room simulation. `s1_anechoic` is the
result of the direct-path simulation and spatial scaling, not the dry WSJ file.
Eight-kHz audio follows the original pipeline (convolution at 16 kHz followed
by resampling); RIRs are resampled separately. Convolving separately resampled
source/RIR is not guaranteed to reproduce that pipeline sample for sample.

For noise-free one-speaker training, select `s1_reverb.scp` as input,
`s1_anechoic.scp` as direct reference, and `rir1_reverb.scp` as the RIR reference.
The manifests share utterance IDs across all signal types. They are signal
indexes, not yet complete ESPnet training data directories with `utt2spk` etc.;
the `rir_1ch` data adapter can use them without fetching RIR NPZ files.

The generator defaults to all combinations and accepts selection arguments:

```bash
bash run_generate_rir_plus.sh --sample-rates 8k 16k --data-lengths min max
# Small check in a new directory:
bash run_generate_rir_plus.sh --splits tt --limit 1 --output-dir /tmp/whamr-plus-check
# Left microphone only:
bash run_generate_rir_plus.sh --mono --output-dir /path/to/new/output
```

Existing output roots are refused, including interrupted runs. Use a fresh
output root; no automatic resume/overwrite is implemented. The existing `data/`
and eight-kHz files are left intact. The default Python environment is `whamr`
(override `WHAMR_PYTHON`), tested with pyroomacoustics 0.3.1. No packages are
installed by the launcher. Scaling metadata is still read from WHAM noise's
original NPZ files during generation; the output RIRs themselves need no NPZ.

Validation used a real test utterance with all four rate/length combinations:
all 15 signal types, stereo/mono correspondence, mixture addition identities,
and agreement with the previous generator for its 13 existing signal types.
Full-dataset generation has not been run as part of this code change.
