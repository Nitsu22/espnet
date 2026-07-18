#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_config=conf/tuning/train_rec_rir_single_clean_8192.yaml
rir_tag=train_rec_rir_single_clean_8192
rir_exp=exp/rir_${rir_tag}
rir_args=

stage=1
stop_stage=6
train_set=tr_mix_clean_reverb_min_8k
valid_set=cv_mix_clean_reverb_min_8k
test_sets=tt_mix_clean_reverb_min_8k
fs=8k
ngpu=1
num_nodes=1
nj=32
dumpdir=dump_rir_clean
audio_format=wav
speech_fold_length=32000
rir_fold_length=8192
skip_data_prep=true
prepare_rec_rir_dump=true
enh_rir_data_dir=../enh_rir/data

. utils/parse_options.sh

if "${prepare_rec_rir_dump}"; then
  local/prepare_rec_rir_dump_rir_clean.sh \
    --dumpdir "${dumpdir}" \
    --enh_rir_data_dir "${enh_rir_data_dir}" \
    --sets "${train_set} ${valid_set} ${test_sets}"
fi

./rir.sh \
  --stage "${stage}" \
  --stop_stage "${stop_stage}" \
  --skip_data_prep "${skip_data_prep}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --fs "${fs}" \
  --ngpu "${ngpu}" \
  --num_nodes "${num_nodes}" \
  --nj "${nj}" \
  --dumpdir "${dumpdir}" \
  --rir_model_type rec_rir \
  --rir_config "${rir_config}" \
  --rir_tag "${rir_tag}" \
  --rir_exp "${rir_exp}" \
  --rir_args "${rir_args}" \
  --speech_fold_length "${speech_fold_length}" \
  --rir_fold_length "${rir_fold_length}" \
  --audio_format "${audio_format}" \
  "$@"
