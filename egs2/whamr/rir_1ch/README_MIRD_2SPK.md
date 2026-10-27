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

Only the reproducible assignment planner is implemented and tested. The measured
MIRD archives have not been acquired, the native MAT fields/angle convention have
not been verified, and audio synthesis has not been implemented or run. An
assignment plan is explicitly marked `awaiting_measured_rir_data`; it is not an
audio dump and must not be used as evidence of completed dataset generation.

Official distribution routes were identified, but the university server timed
out and subsequent external fetches were prevented by automatic approval review
timeouts on 2026-10-27. No source-file substitutes or simulated RIRs were used.

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
implementation hashes. The source-grid angles use the paper's broadside
convention; the native MAT filename/coordinate mapping must be checked before
any audio is generated.

Schema 2 plans label each record `base` or `additional` and identify this sampling
method explicitly. Schema 1 plans from the earlier receiver-based sampler do
not guarantee complete 2,925-case coverage and must not be used for this protocol.
The regenerated production plan is stored separately under
`exp/mird_2spk_preparation_20261027/assignment_plan_2925_plus75/`.
The full WHAMR-ID assignment and twelve unit/integration tests were verified on
2026-10-27; this validation does not include measured RIR files or synthesized
speech audio.
