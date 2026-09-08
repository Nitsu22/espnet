#!/usr/bin/env bash

set -e
set -u
set -o pipefail

./run_rec_rir_tflocoformer_2spk_pit_clean_8192.sh \
  --rir_config conf/tuning/train_rec_rir_tflocoformer_room_path_2spk_pit_clean_8192.yaml \
  --rir_tag train_rec_rir_tflocoformer_room_path_2spk_pit_clean_8192 \
  --rir_exp exp/rir_train_rec_rir_tflocoformer_room_path_2spk_pit_clean_8192 \
  "$@"
