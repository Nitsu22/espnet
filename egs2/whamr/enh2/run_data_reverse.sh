#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Assumes existing /data and /dump already exist
# This script adds reverse speaker position data (reverbse and anechoic_reverse) to existing data

min_or_max=min # "min" or "max". This should match the existing data.
sample_rate=8k

# Use existing dataset names (reverse data will be automatically added via --use_reverse_mix)
train_set=tr_mix_both_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_both_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_both_reverb_${min_or_max}_${sample_rate}"

./enh_reverse.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --use_reverse_mix true \
    --audio_format wav \
    --stop_stage 4 \
    "$@"
