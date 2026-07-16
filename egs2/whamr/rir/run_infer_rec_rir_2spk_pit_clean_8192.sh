#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_exp=exp/rir_train_rec_rir_2spk_pit_clean_8192
train_config=
model_file=
wav_scp=dump_rir_clean/raw/tt_mix_clean_reverb_min_8k/speech_mix_pit.scp
output_dir=
device=cuda
sample_rate=8000
rir_length=8192

. utils/parse_options.sh
. ./path.sh

[ -z "${train_config}" ] && train_config="${rir_exp}/config.yaml"
[ -z "${model_file}" ] && model_file="${rir_exp}/valid.loss.best.pth"
[ -z "${output_dir}" ] && \
  output_dir="${rir_exp}/inference/tt_mix_clean_reverb_min_8k"

if [ ! -f "${wav_scp}" ]; then
  wav_scp=dump_rir_clean/raw/tt_mix_clean_reverb_min_8k/wav.scp
fi

python -m espnet2.bin.rec_rir_pit_inference \
  --train_config "${train_config}" \
  --model_file "${model_file}" \
  --wav_scp "${wav_scp}" \
  --output_dir "${output_dir}" \
  --device "${device}" \
  --sample_rate "${sample_rate}" \
  --rir_length "${rir_length}"
