#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k

train_set=tr_mix_both_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_both_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_both_reverb_${min_or_max}_${sample_rate}"
train_set_reverse=tr_mix_both_reverb_reverse_${min_or_max}_${sample_rate}
valid_set_reverse=cv_mix_both_reverb_reverse_${min_or_max}_${sample_rate}


./se.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --train_set_reverse "${train_set_reverse}" \
    --valid_set_reverse "${valid_set_reverse}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_se_resnet2d.yaml \
    --audio_format wav \
    --stage 6 \
    "$@"
