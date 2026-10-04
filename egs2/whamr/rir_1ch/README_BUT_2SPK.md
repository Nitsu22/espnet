# Two-speaker WHAMR-tt / BUT ReverbDB evaluation

This evaluation set mixes two WSJ speakers convolved with measured BUT RIRs.
It is mono audio with two speakers, not a two-channel recording. The existing
single-speaker `dump_but/` remains unchanged.

From the ESPnet repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python \
  egs2/whamr/rir_1ch/local/create_whamr_but_2spk_dump.py
```

The default output is `egs2/whamr/rir_1ch/dump_but_2spk/`. Generation refuses
an existing output directory. Use `--limit 6 --output PATH` for a small check,
or `--validate-only` to revalidate an existing output. The default is four CPU
workers; no GPU is required. The script estimates storage before generation
and checks that at least 100 GiB remains free during generation.

## Sources and construction

- WHAMR `tt/min`, all 3,000 original mixture IDs, both speakers. The original
  shorter source determines observation length. Eight and sixteen kHz, with
  clean reverberant and WHAM-noisy conditions at each rate.
- Reuse the downloaded RIRs at
  `/net/midgar/work2/nitsu/data/BUT_ReverbDB/rir_metadata/`, the WSJ sources
  under `enh_rir/data/wsj0/wsj0_wav/`, and WHAM noise/scaling metadata under
  `/net/midgar/work2/nitsu/data/wsj/wham_noise/`.
- RIR selection uses the same helper as [the single-speaker set](README_BUT.md):
  microphone 01, MicID01, v00, all source IDs with repeated sessions resolved
  identically. There are 51 RIRs across nine rooms.
- Both sources share the room, receiver setup and microphone coordinates,
  and use distinct measured source coordinates. Geometry is checked explicitly.
- Seed 20260909 preserves the single-speaker set's speaker-1 RIR assignments.
  Speaker-2 partners are balanced per speaker-1 position. The 300 possible
  ordered position pairs within rooms are covered; pair counts within each
  room differ by at most one. Each room has 333 or 334 mixtures.
- RIR assignments are shared across sample rates and noise conditions.
  RIRs retain the distribution's propagation-delay compensation. Resample and
  independently absolute-peak normalize each RIR, retaining signed polarity.
  Convolution uses exactly the coefficients saved in the float32 RIR WAV.
- For each speaker, apply its original WHAM WSJ-mix gain, int16-compatible
  quantization, then the shared WHAM speech gain. Preserve the original WHAM
  noise crop and gain. Noise is WHAM noise, not BUT ambient noise. There is no
  new SIR adjustment after convolution and no fixed SNR target.
- Clip convolved observations to `min` length. Apply one common attenuation
  to all nine signals for each utterance/rate so their peaks do not exceed
  0.99. Clean/noisy conditions share this attenuation. Because speaker 2
  affects the attenuation, speaker-1 waveforms need not be bit-identical to
  the single-speaker dump despite matching its RIR assignment.
- The direct reference is the source convolved with the single signed
  absolute-peak RIR tap. This is an operational reference, not an independently
  verified physical direct-path onset. Original dry sources are also saved.

All selected distributed RIRs are one second long. Keeping their full length
does not recover reverberation beyond one second. This dataset consists of
synthetic mixtures using measured RIRs, not recorded two-speaker conversations.
Use it for held-out evaluation, not training or model selection.

## Saved signals and ESPnet manifests

Four sets are generated directly under `dump_but_2spk/raw/`; no intermediate
Kaldi `data/` directory is required:

```
tt_but_2spk_clean_reverb_min_8k
tt_but_2spk_noisy_reverb_min_8k
tt_but_2spk_clean_reverb_min_16k
tt_but_2spk_noisy_reverb_min_16k
```

Each contains the following manifests. Here `N` means `1` or `2`.

| File | Signal |
| --- | --- |
| `wav.scp`, `speech_mix.scp` | Condition-specific two-speaker input |
| `speech_reverb.scp` | Sum of the two reverberant sources, without noise |
| `speech_reverbN.scp` | Individual reverberant source |
| `speech_directN.scp`, `spkN.scp` | Individual direct reference |
| `sourceN.scp` | Individual dry source, including mixture/common gains |
| `rir_refN.scp`, `rirN.scp` | RIR for that source |
| `noise1.scp` | WHAM noise, including its gain and common attenuation |

`utt2spk1` and `utt2spk2` record actual WSJ speaker IDs. For Kaldi grouping,
`utt2spk` and `spk2utt` map each mixture ID to itself. `utt2num_samples` records
observation lengths. Use `wav.scp` for inputs, including the noisy condition.
There is no ambiguous unnumbered `rir_ref.scp`: evaluation must match both RIRs,
allowing a speaker permutation when the model's output order is unspecified.
Existing one-speaker checkpoints do not directly output this pair of RIRs.

Audio is stored as mono FLOAT WAV under `audio/{8000,16000}/`, with nine signal
directories: `source1`, `source2`, `direct1`, `direct2`, `reverb1`, `reverb2`,
`noise`, `mix_clean`, and `mix_both`. Each rate also has a `rir/` directory.
Clean/noisy manifests share their reference files. `original_rir/` keeps the
selected original RIR files; JSON inventory and metadata record provenance,
geometry, gains, SIR, SNR, paths, and input metadata/RIR hashes.

## Validation

Generation automatically validates all 6,000 utterance/rate tuples: original
source gains and quantization, both source convolutions and direct references,
mixture and noise sums, audio headers and lengths, source/receiver geometry,
rate pairing, RIR hashes, and every manifest mapping. `generation_complete.json`
marks completed generation; `validation.json` is written only after validation
passes. Revalidation removes any previous validation report before checking.

Production log: `exp/but_2spk_generation_20261005/run.log`.

Completed on 2026-10-05: all four sets contain 3,000 mixtures and all 6,000
utterance/rate tuples passed validation. Maximum signal identity error was
`6.497279969597258e-08`. A separate `assignment_audit.json` confirms all 300
ordered pairs, within-room pair counts differing by at most one, distinct WSJ
speakers, and all speaker-1 RIR assignments and source/noise scaling matching
the single-speaker set. The 54,000 signal WAVs total 15,014,812,140 bytes
(about 14.0 GiB); approximately 552.8 GiB remained free after generation.
