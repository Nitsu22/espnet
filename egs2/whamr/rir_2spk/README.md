
## Fixed-RIR paired room/path contrastive learning

`dump_nf_2spk_16k_min_contrastive` contains copied original stereo audio/RIRs
and one fixed, noise-free second mixture for each training/validation item.
Replacement speakers are different from both original speakers and from one
another. The second view reuses the same ordered physical direct/reverb RIRs.
Its source level ratio is independently sampled from [-5, 5] dB; crop offsets,
source identities, waveform hashes, gains and seeds are recorded in
`manifests/{tr,cv,tt}.jsonl`. Synthesis convolves full dry utterances before
cropping. Test audio is copied unchanged and has no second view.
All runtime audio indexes point inside the new dump; historical source paths
in provenance are not dependencies. Storage is approximately 113 GiB.

`local/prepare_room_path_contrastive_dump.py` supports `--estimate-only`,
`--limit` and `--verify-existing`. Use
`local/run_room_path_contrastive_dump_lab.sh FINAL_DUMP RESULT_DIR COMMIT`
on an available lab host to generate with four CPU workers. It checks the
pushed code, creates private `/tmp/nitsu/room-path-paired.*` scratch, runs a
small real-data check, verifies complete generated/copied files, publishes the
dump atomically, and removes its scratch on exit. Logs should be redirected to
RESULT_DIR. Never reuse a directory containing an earlier result.

The model type is `pooled_bimamba_room_path_contrastive` and its preprocessor is
`rec_rir_room_path_contrastive`. The fixed training batch has two RIR-pair
items, expanded to four waveform views inside the model. Each view receives
its own CTF-only PIT assignment before physical-path contrastive matching.
Room positives are the two views of one item; other items supply negatives.
Path positives are the same physical path across views; the other path in
that item supplies negatives. Fixed output-slot embeddings are excluded from
the contrastive features. The baseline padding and CTF/PIM inference remain
compatible.

The initial auxiliary weights are `lambda_room=0.01`, `lambda_path=0.01`,
with temperature 0.1. These are starting settings, not verified optimal values.
Checkpoint selection and early stopping use single-view `valid.loss_sweep`.
Two-view data controls are in
`conf/tuning/train_pooled_bimamba_2spk_nf_16k_room_path_paired_control.yaml`:
its auxiliary weights are zero while both waveform views still receive the
main CTF loss. Compare this control before attributing differences to the
contrastive losses. Room features in this experiment describe a fixed RIR
pair; source-position-independent room disentanglement is not established.

Prepare shapes without starting training:

```bash
bash run_pooled_bimamba_2spk_nf_16k_room_path_contrastive.sh --stage 5 --stop_stage 5
```

Start scratch training after checking an available GPU:

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX bash run_pooled_bimamba_2spk_nf_16k_room_path_contrastive.sh --stage 6 --stop_stage 6
```

For the paired-data control, use the control config and a separate experiment:

```bash
CUDA_VISIBLE_DEVICES=GPU_INDEX bash run_pooled_bimamba_2spk_nf_16k_room_path_contrastive.sh \
  --rir_config conf/tuning/train_pooled_bimamba_2spk_nf_16k_room_path_paired_control.yaml \
  --rir_tag train_pooled_bimamba_2spk_nf_16k_room_path_paired_control
```
