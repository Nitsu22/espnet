#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_exp=exp/rir_train_rec_rir_tflocoformer_2spk_pit_clean_8192
train_config=
model_file=
test_set=tt_mix_clean_reverb_min_8k
wav_scp=
room_param_scp=
output_dir=
score_dir=
device=cuda
sample_rate=8000
rir_length=8192
stage=7
stop_stage=8
skip_inference=false
skip_score=false
snapshot_model=true
snapshot_tag=

. utils/parse_options.sh
. ./path.sh

[ -z "${train_config}" ] && train_config="${rir_exp}/config.yaml"
[ -z "${model_file}" ] && model_file="${rir_exp}/valid.loss.best.pth"
[ -z "${snapshot_tag}" ] && snapshot_tag="$(date +%Y%m%d_%H%M%S)"
[ -z "${wav_scp}" ] && wav_scp="dump_rir_clean/raw/${test_set}/speech_mix_pit.scp"
if [ ! -f "${wav_scp}" ]; then
  wav_scp="dump_rir_clean/raw/${test_set}/wav.scp"
fi
[ -z "${room_param_scp}" ] && \
  room_param_scp="dump_rir_clean/raw/${test_set}/room_param_npz.scp"

if [ ! -f "${train_config}" ]; then
  echo "Missing train config: ${train_config}" >&2
  exit 1
fi
if [ ! -f "${model_file}" ]; then
  echo "Missing model checkpoint: ${model_file}" >&2
  exit 1
fi
if [ ! -f "${wav_scp}" ]; then
  echo "Missing wav scp: ${wav_scp}" >&2
  exit 1
fi

eval_model_file="${model_file}"
eval_train_config="${train_config}"
if "${snapshot_model}"; then
  snapshot_dir="${rir_exp}/eval_snapshots/${snapshot_tag}"
  mkdir -p "${snapshot_dir}"
  cp -p "${train_config}" "${snapshot_dir}/config.yaml"
  cp -p "${model_file}" "${snapshot_dir}/$(basename "${model_file}")"
  eval_train_config="${snapshot_dir}/config.yaml"
  eval_model_file="${snapshot_dir}/$(basename "${model_file}")"
fi

model_name="$(basename "${eval_model_file}" .pth)"
[ -z "${output_dir}" ] && \
  output_dir="${rir_exp}/inference/${test_set}_${model_name}_${snapshot_tag}"
[ -z "${score_dir}" ] && score_dir="${output_dir}/score_rir_pit"

if ! "${skip_inference}" && [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
  python -m espnet2.bin.rec_rir_tflocoformer_pit_inference \
    --train_config "${eval_train_config}" \
    --model_file "${eval_model_file}" \
    --wav_scp "${wav_scp}" \
    --output_dir "${output_dir}" \
    --device "${device}" \
    --sample_rate "${sample_rate}" \
    --rir_length "${rir_length}"
fi

if ! "${skip_score}" && [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
  ./run_score_rec_rir_2spk_pit_clean_8192.sh \
    --pred_dir "${output_dir}" \
    --out_dir "${score_dir}" \
    --sample_rate "${sample_rate}" \
    --rir_length "${rir_length}"
  if [ -f "${output_dir}/rt60_pred.scp" ]; then
    if [ ! -f "${room_param_scp}" ]; then
      echo "Missing RT60 reference scp: ${room_param_scp}" >&2
      exit 1
    fi
    python local/score_rt60_direct.py \
      --prediction_scp "${output_dir}/rt60_pred.scp" \
      --room_param_scp "${room_param_scp}" \
      --output_dir "${score_dir}/rt60_direct"
  fi
fi
