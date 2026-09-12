#!/usr/bin/env bash
set -euo pipefail

# Select a GPU through the caller/scheduler, not a hard-coded device number.
./enh.sh \
    --train_set tr_mix_both_reverb_min_8k \
    --valid_set cv_mix_both_reverb_min_8k \
    --test_sets tt_mix_both_reverb_min_8k \
    --fs 8k --ngpu 1 --ref_num 2 \
    --local_data_opts '--sample_rate 8k --min_or_max min' \
    --enh_config conf/tuning/train_enh_tflocoformer_small_nocashe_bigdeltanet.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_small_nocashe_bigdeltanet \
    --use_dereverb_ref false --use_noise_ref false \
    --inference_model valid.loss.ave_5best.pth \
    --audio_format wav --stage 6 --stop_stage 8 "$@"
