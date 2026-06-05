#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k
rir_input=single_noisy_reverb
rir_length=8192

train_set=tr_rir_${rir_input}_${min_or_max}_${sample_rate}
valid_set=cv_rir_${rir_input}_${min_or_max}_${sample_rate}
test_sets="tt_rir_${rir_input}_${min_or_max}_${sample_rate}"

./rir.sh \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --fs "${sample_rate}" \
  --ngpu 1 \
  --rir_fold_length "${rir_length}" \
  --local_data_opts "--rir_input ${rir_input} --sample_rate ${sample_rate} --min_or_max ${min_or_max} --audio_data_dir ../se2_data/data --npz_data_dir ../se_npz/data --audio_variant rand" \
  --rir_config conf/tuning/train_rir_tflocoformer_small_single_pit_mse_8192.yaml \
  "$@"
