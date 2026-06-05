#!/usr/bin/env bash

set -e
set -u
set -o pipefail

stage=1
stop_stage=8
skip_train=false
skip_inference=false
skip_score=false

rir_config=conf/tuning/train_rec_rir_single_clean_8192.yaml
rir_tag=train_rec_rir_single_clean_8192
rir_exp=exp/rir_${rir_tag}
rir_args=

ngpu=1
num_nodes=1
nj=32
audio_format=wav
device=cuda
inference_model=valid.loss.best.pth
sample_rate=8000
rir_length=8192

train_set=tr_rir_single_clean_reverb_min_8k
valid_set=cv_rir_single_clean_reverb_min_8k
test_set=tt_rir_single_clean_reverb_min_8k

inference_dir=
score_dir=

. utils/parse_options.sh
. ./path.sh

[ -z "${inference_dir}" ] && inference_dir="${rir_exp}/inference/${test_set}"
[ -z "${score_dir}" ] && score_dir="${rir_exp}/score_${rir_length}"

if ! "${skip_train}" && [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 1 ]; then
  train_stage="${stage}"
  train_stop_stage="${stop_stage}"
  if [ "${train_stage}" -lt 1 ]; then
    train_stage=1
  fi
  if [ "${train_stop_stage}" -gt 6 ]; then
    train_stop_stage=6
  fi

  ./run_rec_rir_single_clean_8192.sh \
    --stage "${train_stage}" \
    --stop_stage "${train_stop_stage}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --ngpu "${ngpu}" \
    --num_nodes "${num_nodes}" \
    --nj "${nj}" \
    --audio_format "${audio_format}" \
    --rir_config "${rir_config}" \
    --rir_tag "${rir_tag}" \
    --rir_exp "${rir_exp}" \
    --rir_args "${rir_args}"
fi

if ! "${skip_inference}" && [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
  ./run_infer_rec_rir_single_clean_8192.sh \
    --rir_exp "${rir_exp}" \
    --train_config "${rir_exp}/config.yaml" \
    --model_file "${rir_exp}/${inference_model}" \
    --wav_scp "dump/raw/${test_set}/wav.scp" \
    --output_dir "${inference_dir}" \
    --device "${device}" \
    --sample_rate "${sample_rate}" \
    --rir_length "${rir_length}"
fi

if ! "${skip_score}" && [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 8 ]; then
  ./run_score_rir_single_clean_8192.sh \
    --pred_scp "${inference_dir}/wav.scp" \
    --out_dir "${score_dir}" \
    --sample_rate "${sample_rate}" \
    --rir_length "${rir_length}"
fi

