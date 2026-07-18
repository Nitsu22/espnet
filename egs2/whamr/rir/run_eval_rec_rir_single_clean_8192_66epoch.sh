#!/usr/bin/env bash

set -e
set -u
set -o pipefail

epoch=66
gpu=0
device=cuda
stage=7
stop_stage=8

rir_tag=train_rec_rir_single_clean_8192
rir_exp=exp/rir_${rir_tag}
dumpdir=dump_rir_clean
test_set=tt_mix_clean_reverb_min_8k
sample_rate=8000
rir_length=8192

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

. utils/parse_options.sh

model_name="${epoch}epoch.pth"
model_file="${rir_exp}/${model_name}"

if [ ! -f "${model_file}" ]; then
  echo "Missing model checkpoint: ${model_file}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu}"

inference_dir="${rir_exp}/inference/${test_set}_${epoch}epoch"
score_dir="${rir_exp}/score_${rir_length}_${epoch}epoch"

./run_rec_rir_single_clean_8192_all.sh \
  --stage "${stage}" \
  --stop_stage "${stop_stage}" \
  --skip_train true \
  --prepare_rec_rir_dump false \
  --inference_model "${model_name}" \
  --inference_dir "${inference_dir}" \
  --score_dir "${score_dir}" \
  --device "${device}" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}" \
  --dumpdir "${dumpdir}" \
  --test_set "${test_set}"
