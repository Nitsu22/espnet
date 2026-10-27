# MIRD / WHAMR two-speaker evaluation preparation

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

The downloader and native archive/metadata validator are implemented. Audio
synthesis has not been implemented or run; no completed audio dump is claimed.
Assignment plans remain marked `awaiting_measured_rir_data`. ZIP integrity alone
does not validate a recording condition; the downloader now fails when native
condition validation fails. An older `download_complete.json` is only a transport
record, not evidence of a complete usable MIRD database.

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
The full WHAMR-ID assignment, nineteen unit/integration tests, and all eight valid
native recording conditions were verified on 2026-10-27. The source audit exits
with failure because the ninth condition is missing. No synthesized speech
audio was validated.

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
