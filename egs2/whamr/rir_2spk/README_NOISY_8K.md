# Noisy WHAMR: native 8 kHz BiMamba CTF estimator

This is a new scratch-trained estimator, not the downstream separator. Its YAML
is identical to the existing native 8 kHz NF BiMamba YAML. The only experimental
change is observing `mix_both_reverb` instead of `mix_clean_reverb`. Two clean
speaker-specific direct/reverberant RIR teachers and all utterance IDs stay fixed.
Noise is never a teacher or inference input. The predictor sees the left channel.

Settings: 439,532 parameters, pre/post 4/2, batch basis 4, validation batch 1,
4-second training crop, FP32, seed 0, SweepCTF v2 RIMag/PIT, 60 taps, FFT/window/hop
256/256/128, AdamW and cosine restarts unchanged, at most 100 epochs and the
existing patience 10. Fold threshold 80,000 preserves the NF estimator's batching
policy (this is not the separator's fixed-batch policy).

The 20,000/5,000/3,000 train/validation/test waveforms already exist under
`data_rir_plus`. Preparation copies only indexes and metadata, never audio/RIRs.
All headers are checked; the noisy/NF audit checks every ID, speaker, teacher,
length and output checksum, and samples each split's nonzero noise and mixture sum.

```bash
bash run_pooled_bimamba_2spk_noisy_8k_sweep_v2.sh \
  --stage 1 --stop_stage 5 --python /path/to/python \
  --source_data /absolute/data_rir_plus --dumpdir /absolute/new_noisy_dump \
  --rir_stats_dir /absolute/new_stats
python local/verify_noisy_two_speaker_dump.py --noisy /absolute/new_noisy_dump \
  --clean /absolute/existing_nf_8k_dump --output /absolute/data_check.json
```

The lab launcher additionally verifies an idle GPU, a real FP32 batch-four update,
finite gradients, speaker-teacher PIT invariance, two-RIR inference and full-length
validation including the longest utterance. It does not install dependencies,
change the batch after OOM, warm-start from the clean checkpoint, or submit any
downstream separation experiment. Launch it in a detached tmux session. It saves
the host, Git commit, GPU, probe, training log and final exit status. Resume using
the run script with the same paths and `--resume true --stage 6 --stop_stage 6`.
