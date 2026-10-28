# Frozen BiMamba CTF-input controls (NF-WHAMR 8 kHz / Concat S)

Evaluation only. No model, training, dataset/loader or shared inference API
changes. Use the same immutable Concat S 129epoch checkpoint, whose frozen
BiMamba predictor is embedded and SHA-pinned.

The separation mixture and both scoring references are ALWAYS the original
3,000 test examples, channel 0. Only the waveform passed to frozen BiMamba changes:

- same_rir_other_utt: s1_prime*h1 + s2_prime*h2, with two different test
  utterances (neither equals either original utterance), length preserved.
- same_utt_swapped_rir: s1*h2 + s2*h1, with original utterances and exchanged
  RIR assignments. This retains the SAME RIR SET. It is not a wrong-RIR-set
  negative control; interpret it as speech/RIR-assignment sensitivity.
- same_utt_other_rir: s1*h3 + s2*h4, retaining the ORIGINAL utterances but
  selecting both physical RIRs from ONE OTHER original test scene. A seed-0
  Sattolo derangement over all 3,000 sorted test UIDs gives no fixed points and
  uses every donor pair once. Even a swapped hash-identical reverberant pair
  is rejected. This is the RIR-set-mismatch negative control.

The generator defaults remain the first TWO controls. Select only the new
control with --conditions same_utt_other_rir. It does not select any speech
donors. Pilot selection is applied AFTER building the full RIR derangement,
so the pilot and full mappings agree. Source power matching, shared
direct/reverberant gains, common attenuation, original duration and PCM16
format are unchanged. Audit records actual/original physical RIR paths and
hashes, donor_rir_uid, target_t60 and source_t60 (nominal donor-scene RT60).
Nominal RT60 need not differ; the physical RIR pair must differ by hashes.

The existing standard Baseline and matched-CTF scores are read-only comparison
inputs. They are not recomputed or overwritten. No checkpoint reselection.

## Preparation / transfer

The CPU generator and audits are in prepare_nf8k_ctf_controls.py. Transfer the
entire completed dump without rewriting its indexes or generation manifest.
The evaluation prepare mode verifies every generation checksum, the original
mixture/reference hashes recorded in the generation audit, UID sets, durations,
RIR assignment and existing scores. It writes separate, rebased conditioning
indexes; the transferred source indexes/audits remain byte-identical.

## Inference guard

ctf_audio_intervention.py uses a temporary PyTorch pre-hook on frozen_ctf.
It does not modify the waveform passed to the separation encoder, weights,
configuration, normalization or output writer. Hooks are removed on failure.

Inference is full-waveform float32, batch 1, 8 kHz, seed 0,
normalize_output_wav=true. Before any full test output is created, the shortest,
median-length and longest scenes must pass:

1. Standard inference versus intervention with the ORIGINAL estimator input:
   max waveform error <= 1e-5.
2. Standard inference versus previously saved normalized matched outputs:
   max waveform error <= 5e-5 (includes PCM16 quantization).
3. Both real interventions produce finite correctly sized outputs and
   [1, 60, 2, 129, 2] CTFs.

Failure stops full inference and the dependent score job. smoke.json records
the controls, not a performance-based scene/checkpoint selection.
Each full scene records original/conditioning paths and estimated CTF hash.

## TSUBAME

Submit from enh_rir in a dedicated synchronized checkout. Use paid group jobs,
priority -5. Do not run inference/scoring on a login node.

- prepare: cpu_4=1, h_rt=00:30:00.
- infer: gpu_1=1, h_rt=01:00:00; hold_jid prepare; pass predecessor job ID.
- score: cpu_4=1, h_rt=01:00:00; hold_jid infer; pass predecessor job ID.

Script: qsub/eval_ctf_audio_controls.sh MODE OUTPUT [PREDECESSOR_ID].
For prepare only, the optional third argument is instead an absolute source
dump root. For the single wrong-RIR condition, the current resource/time
plan is cpu_4/15 min preparation, gpu_1/30 min inference, cpu_4/30 min scoring.
CPU audio generation is local/run_prepare_nf8k_wrong_rir_cpu.sh RECIPE JOBDIR
on a lab CPU; it runs a 10-scene pilot and then the complete 3,000 scenes.
Dependents require predecessor exit_status=0 and identical Git commit, not
merely scheduler completion. Original data and model paths are explicit there.

Scoring uses the standard ESPnet enh_scoring with four single-threaded shards,
original references and ref_channel=0. All keys and finite metrics are checked.
summary.csv contains SI_SNR (zero-mean SI-SDR), SDR, SAR, SIR, STOI for
Baseline, matched and both interventions, including paired mean differences.
ESTOI is additionally saved for the new conditions, but is not compared to the
historical Baseline because that experiment did not save ESTOI.
paired_si_snr.csv preserves scene-level values. score_complete.json is the
success marker. Partial outputs are never silently reused/overwritten.

CPU fixture tests: test/espnet2/enh/test_ctf_audio_intervention.py.
