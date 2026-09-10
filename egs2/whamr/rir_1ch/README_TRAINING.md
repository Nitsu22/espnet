# Scratch single-speaker Rec-RIR

`data_rir_plus` is a relative symlink to `../enh_rir/data_rir_plus`.
The source stereo WAVs are shared. No waveform copies are made in the dump.

Defaults: NF-WHAMR, speaker 1, left microphone (index 0), 16 kHz, min,
20,000 train / 5,000 validation / 3,000 test entries. Input and reverberant
reference both use `s1_reverb`. Direct reference uses `s1_anechoic` (the spatial
direct-path simulation, not dry WSJ). RIR references use `rir1_reverb` and
`rir1_anechoic`. ACE is evaluation only.

## Prepare and train

From this recipe, in the existing `tf-locoformer` environment:

```bash
bash run_rec_rir_single_nf_16k.sh --stage 1 --stop_stage 1
# On an available lab GPU, inside tmux and following the lab GPU skill:
bash run_rec_rir_single_nf_16k.sh --stage 5 --stop_stage 6
# Explicit restart from this experiment's checkpoint:
bash run_rec_rir_single_nf_16k.sh --stage 6 --stop_stage 6 --resume true
```

Stage 1 creates `dump_nf_16k_min/raw/{tr,cv,tt}_rir_single_nf_min_16k` and
checks all selected audio/RIR headers and IDs. Identical prepared indexes can
be reused; changed source manifest hashes, settings or output indexes fail.
Stages 5/6 use the existing `rir.sh` statistics/training implementation.
Stats are isolated by `rir_tag`. Default experiment:
`exp/rir_train_rec_rir_single_nf_16k`.

The YAML explicitly has `init_param: []`; the launcher defaults to
`resume=false` and refuses an existing training checkpoint unless resuming is
explicit. No released weights are loaded. The model uses FFT/window 512,
hop 256, 257 frequency bins, 60 CTF taps and 6/2/6 speech/noise/CTF layers.
Training crops aligned audio triples to at most 4 seconds; validation uses
full utterances, as in the existing ESPnet preprocessor. Batch size 4, FP32,
AdamW lr=0.001, all three loss weights 1, max 100 epochs, early stopping
patience 10. Scheduler first period is 5,000 optimizer updates. These are
the stated ESPnet experiment settings, not an exact official training reproduction.

`force_single_channel: true` selects left-channel audio in preprocessing.
The dump `channel` file records the convention; it is metadata, not an automatic
channel selector for arbitrary models. A different model consuming stereo RIR
WAVs must also select channel 0 explicitly. The indexes are model-independent,
but new networks/losses may need their own task adapters; they are not claimed
to run automatically through the legacy NPZ-based direct-RIR task.

## Evaluate

```bash
bash run_eval_rec_rir_single_nf_16k.sh --dataset whamr
bash run_eval_rec_rir_single_nf_16k.sh --dataset ace_clean
bash run_eval_rec_rir_single_nf_16k.sh --dataset ace_noisy
```

The default checkpoint is `valid.loss.best.pth`. The existing ESPnet inference
selects the left microphone. Evaluation selects left-channel reference RIRs,
uses 32000-sample predictions and the same peak-aligned two-second protocol
as the released Rec-RIR ACE evaluation. Existing evaluation output directories
are refused. `--model_file`, `--rir_exp`, `--output_dir` and `--device` are
configurable. Scores are this repository's protocol, not official SimACE scores.

For a bounded GPU check, `local/check_rec_rir_training.py` performs one random
initialization, forward/backward at the configured precision and optimizer step, then RIR inference.
It does not start a full training experiment or load a pretrained checkpoint.

Verified on 2026-09-10: all 28,000 selected entries passed header/ID checks;
repeat preparation verified the saved index hashes. Statistics collection on
two train and two validation entries passed (local sandbox check used zero
workers). On aurum RTX 3090 / PyTorch 2.1.0+cu118, the scratch FP32 check used
64,000 input samples, produced finite loss 15.3129959 and finite gradients,
updated parameters and inferred 32,000 RIR samples. The report is
`exp/scratch_training_check/check.json`. Initial AMP testing produced nonfinite
gradients, so the default is FP32. A stereo-reference/mono-prediction identity
test yielded zero RMSE and zero acoustic-parameter errors with channel 0.
Full training has not been started by these checks.
