# Single-speaker WHAMR-tt / BUT ReverbDB evaluation

Raw archive: /net/midgar/work2/nitsu/data/BUT_ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz.
RIRs/metadata only are extracted into that directory's rir_metadata/ subdirectory.
The archive passed gzip integrity checking; SHA256SUMS records its locally
computed digest (not an independently published publisher checksum).
Official source: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database

Generate using the tf-locoformer environment's Python from the ESPnet root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python \
  egs2/whamr/rir_1ch/local/create_whamr_but_dump.py
```

The script generates dump_but/ directly in rir_1ch. No intermediate Kaldi data/
is required. Original downloaded data remains in /net/midgar/work2/nitsu/data/.
The output must not already exist. --limit 3 --output PATH is a smoke check;
--validate-only rechecks an existing dump. --extract-only safely extracts only
RIRs and text metadata into a new --rir-root from the adjacent archive.

Protocol:

- WHAMR tt/min, 3000 original IDs, speaker 1 only; speaker 2 sets min length.
- Eight and sixteen kHz, mono, noise-free reverberant and WHAM-noisy inputs.
- User-selected microphone 01 in MicID01 in each room. Use all source IDs,
  one v00 RIR each; repeated source IDs prefer dedicated RIR (_S) sessions,
  then earliest session. Actual selected members and coordinates are recorded.
- Balance utterances over rooms, then over source positions within each room.
  Seed 20260909. Assignments are shared across rates and noise conditions.
- Preserve distributed BUT RIRs, including their existing propagation-delay
  compensation. No acquisition-delay restoration is attempted. This differs
  from the ACE archive, whose acquisition delay was retained.
- Full RIRs are resampled and absolute-peak normalized. Keep signed polarity,
  full length and exactly the saved float32 convolution coefficients.
- Direct reference uses the single signed peak tap convolved with the source;
  it is an operational reference, not a physical-onset guarantee.
- Reuse ACE generator's WSJ/WHAM scaling, quantization, noise crop and gains.
  Convolve before clipping observed audio to min length. Common attenuation
  applies to all saved signals and is shared by clean/noisy conditions.
- WHAM noise is used, not BUT ambient noise. SNR is not fixed to 20 dB.
- Test only: do not use for training or model selection. This is not SimACE.
- Reserve 100 GiB free; require another 20 GiB before generation starts.

Outputs mirror dump_ace:

- raw/tt_but_single_{clean,noisy}_reverb_min_{8k,16k}/: wav.scp,
  speech_mix.scp, speech_reverb.scp, speech_direct.scp, spk1.scp,
  source.scp, rir_ref.scp, rir.scp, speaker and length maps.
- audio/{8000,16000}/{source,direct,reverb,noise,mix,rir}/: FLOAT mono WAVs.
- original_rir/: byte-identical selected original RIRs.
- rir_inventory.json: source members, room, microphone/source IDs, metadata.
- generation_config.json, rir_metadata.json, utterances.jsonl: provenance.
- generation_complete.json: written when generation finishes.
- validation.json: written only after all signal/SCP checks pass.

Use wav.scp for model input (especially noisy sets); speech_reverb.scp is always
noise-free. Use rir_ref.scp for common RIR scoring, matching the ACE protocol.

The downloaded archive supplies 51 selected RIRs across 9 rooms after removing
repeated source IDs: ConferenceRoom2 4, Room112 5, C236 10, D105 6, E112 2,
L207 6, L212 5, L227 10, Q301 3. The archive has more L227 source IDs than the
summary table on the website; inventory files, not that table, define this set.
All selected distributed RIRs are mono 16 kHz and exactly 1 second long.
Keeping the full distributed RIR therefore does not recover reverberation
beyond one second; a two-second evaluation window will pad that reference.
Their absolute peaks are at samples 32 through 2534 at 16 kHz. We do not
interpret those peaks as independently verified physical direct-path onsets.

Completed and validated on 2026-09-15: all four sets contain 3000 utterances.
The 6000 rate/utterance tuples passed waveform, convolution, direct-reference,
additive-noise, RIR hash/header, rate-pairing and SCP checks. Maximum signal
identity error: 7.358777009969231e-08. Room counts are 333 or 334 each.
Generated speech audio totals 8,341,562,300 bytes (shared references are not
duplicated between clean/noisy conditions). Logs: exp/but_generation_logs/run.log.
The three-example smoke set is under exp/but_generation_smoke/.
Recorded microphone coordinates agree within each room, and repeated source
IDs have matching source/microphone coordinates. Full room/source assignment
balance and the ACE-compatible manifest schema were also checked.
