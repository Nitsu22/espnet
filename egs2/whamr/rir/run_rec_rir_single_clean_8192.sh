#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_config=conf/tuning/train_rec_rir_single_clean_8192.yaml
rir_tag=train_rec_rir_single_clean_8192
rir_exp=exp/rir_${rir_tag}

./rir.sh \
  --train_set tr_rir_single_clean_reverb_min_8k \
  --valid_set cv_rir_single_clean_reverb_min_8k \
  --test_sets tt_rir_single_clean_reverb_min_8k \
  --fs 8k \
  --ngpu 1 \
  --rir_model_type rec_rir \
  --local_data_opts "--rir_input single_clean_reverb --sample_rate 8k --min_or_max min" \
  --rir_config "${rir_config}" \
  --rir_tag "${rir_tag}" \
  --rir_exp "${rir_exp}" \
  --speech_fold_length 32000 \
  --rir_fold_length 8192 \
  --audio_format wav \
  "$@"

