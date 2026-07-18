#!/usr/bin/env bash

set -e
set -u
set -o pipefail

rir_config=conf/tuning/train_rec_rir_2spk_pit_clean_8192_loss_rec_only.yaml
rir_tag=train_rec_rir_2spk_pit_clean_8192_loss_rec_only
rir_exp=exp/rir_${rir_tag}

./run_rec_rir_2spk_pit_clean_8192.sh \
  --rir_config "${rir_config}" \
  --rir_tag "${rir_tag}" \
  --rir_exp "${rir_exp}" \
  "$@"
