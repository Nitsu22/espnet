#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_exp=exp/rir_train_rec_rir_2spk_pit_clean_8192
pred_dir=
pred_rir_scp=
pred_rir1_scp=
pred_rir2_scp=
out_dir=
ref_rir_scp=dump_rir_clean/raw/tt_mix_clean_reverb_min_8k/rir.scp
sample_rate=8000
rir_length=8192
align=peak
scale_mode=peak
direct_window_ms=2.5
pit_metric=rmse_50ms

. utils/parse_options.sh
. ./path.sh

[ -z "${pred_dir}" ] && pred_dir="${rir_exp}/inference/tt_mix_clean_reverb_min_8k"
[ -z "${out_dir}" ] && out_dir="${pred_dir}/score_rir_pit"

if [ ! -f "${ref_rir_scp}" ]; then
  echo "Missing reference RIR scp: ${ref_rir_scp}" >&2
  exit 1
fi

pred_opts=
if [ -n "${pred_rir_scp}" ]; then
  pred_opts="--pred_rir_scp ${pred_rir_scp}"
elif [ -n "${pred_rir1_scp}" ] || [ -n "${pred_rir2_scp}" ]; then
  if [ -z "${pred_rir1_scp}" ] || [ -z "${pred_rir2_scp}" ]; then
    echo "Both --pred_rir1_scp and --pred_rir2_scp are required together" >&2
    exit 2
  fi
  pred_opts="--pred_rir1_scp ${pred_rir1_scp} --pred_rir2_scp ${pred_rir2_scp}"
else
  pred_opts="--pred_dir ${pred_dir}"
fi

local/score_rir_pit.py \
  --ref_rir_scp "${ref_rir_scp}" \
  ${pred_opts} \
  --out_dir "${out_dir}" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}" \
  --align "${align}" \
  --scale_mode "${scale_mode}" \
  --direct_window_ms "${direct_window_ms}" \
  --pit_metric "${pit_metric}"
