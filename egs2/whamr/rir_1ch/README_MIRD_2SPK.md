# MIRD / WHAMR two-speaker evaluation generation

Target: the same 3,000 WHAMR `tt/min` source mixtures used by the BUT evaluation,
with MIRD measured RIRs, at 8/16 kHz and clean/WHAM-noisy conditions. Both speakers
must share the RT setting, array configuration and microphone, and occupy
different measured source positions. This is a single-channel synthetic mixture
using measured RIRs, not recorded two-speaker speech.

MIRD contains one physical room with RT settings 160/360/610 ms, three eight-mic
array configurations, and 26 source positions (13 angles at 1 m and 2 m).
The 72 receiver recording conditions include overlapping physical microphone
positions. The 23,400 unordered receiver/source-pair combinations are therefore
not 23,400 independent rooms or physical geometries.

[Source paper](https://sharongannot.group/wp-content/uploads/2021/01/Hadad-Heese-Vary-Gannot-MULTICHANNEL-AUDIO-DATABASE-IN-VARIOUS-ACOUSTIC-ENVIRONMENTS.pdf).

## Current status

The 3,000-mixture assignment is prepared. All nine official ZIPs and the SP tools
were downloaded on 2026-10-27 from the public Google Drive folder linked by the
[author's downloads page](https://sharongannot.group/resources/downloads/).
Eight recording conditions passed verification of all 208 native MAT files.
Each contains a 480,000 × 8 floating-point RIR at 48 kHz (ten seconds).

**Generation is blocked by a mispackaged official archive.** The ZIP labeled
`Reverberation_0.610s)_3-3-3-8-3-3-3.zip` contains only another ZIP, labeled
`Reverberation_0.360s)_8-8-8-8-8-8-8.zip`. The inner ZIP is byte-identical to the
separately downloaded 360-ms / 8-cm archive. These are locally computed SHA256s:

- Incorrect outer archive:
  `0431db78c6718b46c6a42d8decd244f2a8bb63389935f05c389bdbc28a8863c1`
- Inner archive and the separate 360-ms / 8-cm archive:
  `c24f10845b2624648868faf5153a42e0a19d83bdd8986c86d10e7bc62a05231a`

The university's old distribution server was unreachable. A FullSubNet research
mirror was also inspected, but it contains only 119 of the required 208 mono
responses for the missing condition, at 16 kHz / PCM16 rather than native MATs.
It was not used as a replacement. The user has no separate native copy.

The missing session affects 334 assigned mixtures (325 base cases plus nine
additions). Completing the approved nine-condition protocol requires the correct
26 native MATs for 610 ms / `3-3-3-8-3-3-3`, including all eight microphones.
Changing to eight conditions would change the approved coverage design.

The downloader, native archive/metadata validator, audio generator and independent
audio validator are implemented. An eight-mixture smoke run covers every valid
recording condition, at both sample rates and with both clean/noisy mixtures.
This is **not the production 3,000-mixture dataset**. Full production generation
rejects incomplete native conditions before creating its output directory.

Assignment plans remain marked `awaiting_measured_rir_data` until all nine native
conditions can be used. ZIP integrity alone does not validate a recording
condition; the downloader fails when native condition validation fails. An older
`download_complete.json` is only a transport record, not evidence of a complete
usable MIRD database.

Detailed source verification is saved locally in
`exp/mird_2spk_preparation_20261027/native_rir_audit.json`, with all 208 checked
recordings, archive hashes and the failing archive's nested-ZIP fingerprint.

## Assignment protocol

- Cover all **325 unordered source pairs × 3 RT settings × 3 array
  configurations = 2,925 base cases**, once each. Select one microphone for
  each case using seeded, balanced random assignment.
- Add **75 cases**, with distinct source-pair types globally. Each adds a second
  occurrence of one base case at a different microphone. The remaining 2,850
  base cases occur only once; all 3,000 full receiver/source-pair conditions
  are distinct.
- Exactly 1,000 mixtures per RT setting and per array configuration.
- Exactly 375 mixtures per microphone number across RT/array settings.
- Each RT/array session has 333 or 334 mixtures: 325 base cases plus 8 or 9
  additions. Each RT setting and each array has exactly 25 additions.
- All 72 receiver recording conditions: 41 or 42 mixtures each.
- No repeated unordered source pair at any one receiver condition.
- Each source-pair type appears in all nine RT/array sessions. The 75 additional
  pair types occur ten times overall; the remaining 250 occur nine times.
- Source-to-speaker assignments are balanced globally: each of the 26 positions
  occurs 115 or 116 times on each speaker side. This is not a per-receiver
  source-role guarantee.
- Microphone selection is constrained random sampling, not independent random
  draws: per-session channel quotas are fixed, and an added case cannot reuse
  its base case's microphone. Array layouts retain their recorded identities
  even where physical microphone positions overlap.
- Keep original WHAMR source IDs and their scaling-record order. Later synthesis
  must reuse the same assignment across sample rates and noise conditions.

From the repository root, using the existing ESPnet Python environment:

```bash
python egs2/whamr/rir_1ch/local/mird_pair_assignment.py \
  --filenames egs2/whamr/enh_rir/whamr_scripts/data/mix_2_spk_filenames_tt.csv \
  --scaling /net/midgar/work2/nitsu/data/wsj/wham_noise/metadata/scaling_tt.npz \
  --output egs2/whamr/rir_1ch/exp/mird_assignment_plan_complete_cases

python -m unittest discover -s test/espnet2/rir -p test_mird_pair_assignment.py
```

Plans refuse an existing output directory and record source-index and assignment
implementation hashes. The official SP package's `sample_create_soundfield.m`
and native MAT metadata establish the angle convention: 0° is front, 90° is left
(toward mic1), and 270° is right (toward mic8). Plans record the native azimuth as
`angle_deg % 360`, retaining the signed source-grid order. No signal is trimmed,
resampled or normalized by the source validator.

Schema 2 plans label each record `base` or `additional` and identify this sampling
method explicitly. Schema 1 plans from the earlier receiver-based sampler do
not guarantee complete 2,925-case coverage and must not be used for this protocol.
The regenerated production plan is stored separately under
`exp/mird_2spk_preparation_20261027/assignment_plan_2925_plus75/`; a new plan with
verified native angles is in `assignment_plan_native_angles/` alongside it.
The full WHAMR-ID assignment, 31 unit/integration tests, and all eight valid
native recording conditions were verified on 2026-10-27. The source audit exits
with failure because the ninth condition is missing.

To rerun native verification without extracting or modifying the archives:

```bash
python egs2/whamr/rir_1ch/local/mird_rir.py \
  --archives egs2/whamr/rir_1ch/downloads/MIRD/archives \
  --report egs2/whamr/rir_1ch/exp/mird_native_audit_new.json

python -m unittest discover -s test/espnet2/rir -p 'test_mird*.py'
```

The report refuses an existing output file. `download_mird.py` accepts the saved
public Drive inventory (`--inventory`) and a destination (`--output`); it verifies
ZIP CRCs, computes transport hashes, then runs the same native metadata audit.
It uses no account cookies or user cache. No downloaded MATLAB code is executed.

## Audio protocol and verified smoke output

`local/create_whamr_mird_2spk_dump.py` reads the immutable complete assignment.
Production uses all 3,000 entries; `--smoke-per-condition N` selects the first N
planned entries for each available RT/array session, retaining their original
WHAM source IDs, plan indices, source positions and microphone. Smoke selection
does not redraw or repair missing assignments.

- Select the same recorded microphone column for both source positions.
- Keep the full native ten-second RIR, including its delay. Resample to 8/16 kHz,
  divide each mono RIR by its absolute peak, and store FLOAT WAV references.
  This peak normalization follows the BUT evaluation convention; absolute
  measured distance attenuation is not preserved.
- Resample the original WSJ speech and crop to the shorter source. Apply each
  original WHAM WSJ gain, MATLAB-compatible int16 quantization, then the shared
  WHAM speech gain. Convolve with the complete RIR and crop to the speech length.
- Construct direct references from one signed absolute-peak tap of each RIR.
  This is a peak-tap target, not an independently verified physical direct onset.
- Use the original WHAM noise gain and start index, with the left-channel mono
  convention. Apply one common attenuation to all nine outputs: `source1/2`,
  `direct1/2`, `reverb1/2`, `noise`, `mix_clean`, and `mix_both`.
- Write four ESPnet SCP sets, one per sample rate and clean/noisy condition.
  Preserve the same assignment across all four. No SIR/SNR is forced after
  measured-RIR convolution.

The actual smoke result is
`dump_mird_2spk_smoke_native10s_20261027_v2/`: eight mixtures in each of four
sets, 16 rate/utterance records, and 32 saved mono RIRs. Independent reconstruction
from the native MATs and original CSV/NPZ/audio gave zero RIR error and maximum
signal error `5.78e-8` (FLOAT rounding; tolerance `3e-6`). The validator checks the
selected microphone, both convolutions/direct targets, source/noise gains,
independently recomputed common attenuation, mixture sums, paired sample rates,
archive hashes and all SCP mappings.

Smoke outputs have `smoke_complete.json` and a `validation.json` explicitly marked
`is_production_dataset: false`. Production uses `generation_complete.json`, then
requires the independent validation result `validated_production`. Neither a
smoke marker nor an assignment plan is a production completion claim.

Example from a checkout with the raw assets available (use explicit asset paths
when running from an isolated worktree):

```bash
python egs2/whamr/rir_1ch/local/create_whamr_mird_2spk_dump.py \
  --output egs2/whamr/rir_1ch/dump_mird_smoke_new \
  --plan egs2/whamr/rir_1ch/exp/mird_2spk_preparation_20261027/assignment_plan_native_angles/assignment_plan.json \
  --archives egs2/whamr/rir_1ch/downloads/MIRD/archives \
  --audit-report egs2/whamr/rir_1ch/exp/mird_2spk_preparation_20261027/native_rir_audit.json \
  --smoke-per-condition 1 --workers 2

python egs2/whamr/rir_1ch/local/create_whamr_mird_2spk_dump.py \
  --output egs2/whamr/rir_1ch/dump_mird_smoke_new --validate-only
```

Use a fresh output directory. Omitting `--smoke-per-condition` selects production
and requires the correct native 610-ms / 3-cm archive. If any native archive is
replaced, rerun the complete source audit with a new report path first. Cached
audit reuse always verifies all archive hashes against the saved report.
