#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_exp=exp/rir_train_rec_rir_2spk_pit_clean_8192
model_file="${rir_exp}/36epoch.pth"
output_dir="${rir_exp}/inference/tt_mix_clean_reverb_min_8k_36epoch"
device=cuda
sample_rate=8000
rir_length=8192

. utils/parse_options.sh

if [ ! -f "${model_file}" ]; then
  echo "Missing model checkpoint: ${model_file}" >&2
  exit 1
fi

./run_infer_rec_rir_2spk_pit_clean_8192.sh \
  --rir_exp "${rir_exp}" \
  --model_file "${model_file}" \
  --output_dir "${output_dir}" \
  --device "${device}" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}"

./run_score_rec_rir_2spk_pit_clean_8192.sh \
  --rir_exp "${rir_exp}" \
  --pred_dir "${output_dir}" \
  --out_dir "${output_dir}/score_rir_pit" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}"
