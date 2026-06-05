#!/usr/bin/env bash

set -e
set -u
set -o pipefail

pred_dir=
pred_scp=
out_dir=
ref_rir_npz_scp=data/tt_rir_single_clean_reverb_min_8k/rir_npz.scp
sample_rate=8000
rir_length=8192
source_index=0
mic_index=0
align=peak
scale_mode=peak
direct_window_ms=2.5

. utils/parse_options.sh
. ./path.sh

if [ -z "${out_dir}" ]; then
  echo "--out_dir is required" >&2
  exit 2
fi
if [ -z "${pred_dir}" ] && [ -z "${pred_scp}" ]; then
  echo "Either --pred_dir or --pred_scp is required" >&2
  exit 2
fi
if [ -n "${pred_dir}" ] && [ -n "${pred_scp}" ]; then
  echo "Use only one of --pred_dir or --pred_scp" >&2
  exit 2
fi
if [ ! -f "${ref_rir_npz_scp}" ]; then
  fallback_ref=../se_npz/data/tt_mix_single_reverb_min_8k/rir_npz.scp
  if [ -f "${fallback_ref}" ]; then
    ref_rir_npz_scp="${fallback_ref}"
  else
    echo "Missing reference RIR scp: ${ref_rir_npz_scp}" >&2
    exit 1
  fi
fi

pred_opts=
if [ -n "${pred_dir}" ]; then
  pred_opts="--pred_dir ${pred_dir}"
else
  pred_opts="--pred_scp ${pred_scp}"
fi

local/score_rir.py \
  --ref_rir_npz_scp "${ref_rir_npz_scp}" \
  ${pred_opts} \
  --out_dir "${out_dir}" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}" \
  --source_index "${source_index}" \
  --mic_index "${mic_index}" \
  --align "${align}" \
  --scale_mode "${scale_mode}" \
  --direct_window_ms "${direct_window_ms}"
