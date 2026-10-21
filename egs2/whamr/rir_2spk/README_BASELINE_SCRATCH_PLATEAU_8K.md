# BiMamba Baseline 8 kHz: scratch training with plateau decay

This experiment uses **random initialization**, `init_param: []`, and
`resume: false` in a new output directory. No epoch-45, fine-tuned, or other
pretrained checkpoint is loaded. The GPU profiling updates are discarded;
Stage 6 builds a fresh model using seed 0.

- Initial LR: **0.001**, AdamW, weight decay 0, no warm restarts.
- Maximum: **200 epochs**.
- Halve LR after **five consecutive validation epochs without improvement**.
  PyTorch `ReduceLROnPlateau.patience=4` implements five bad epochs, since
  the reduction condition is `num_bad_epochs > patience`. Floor: 1e-6.
- Preserve the previous early-stopping policy: stop after 15 bad epochs
  (`patience=14` in ESPnet); 200 is a maximum, not a guaranteed duration.
- Same 439,532-parameter two-speaker model, spectral SweepCTF v2 PIT loss,
  NF 8 kHz dump, seed, batch size 4, input-only folded batching, 4-second
  training crops, full-length validation with batch size 1, and no AMP.
- Select by `valid.loss`, retain the best five checkpoints. Reject
  all-invalid-gradient termination even if ESPnet returns exit code zero.

Launch on a verified free lab GPU:

```bash
bash local/run_baseline_scratch_plateau_8k_lab.sh 0 \
  /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
```

Configuration:
`conf/tuning/train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep.yaml`.
Model results:
`exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep`.
Control and verification:
`exp/baseline_scratch_plateau_200ep_training_20261022`.

The launcher checks scratch initialization, model/data parity, the actual
library's five-bad-epoch decay, manifest hashes, finite GPU updates, paired
PIT invariance, and two-RIR output. It refuses existing result directories
and checks GPU availability again before Stage 6. Cache and temporary
files use the owned run directory and are removed at process exit.
