#!/usr/bin/env bash

set -e
set -u
set -o pipefail

./run_eval_rec_rir_tflocoformer_2spk_pit_clean_8192_current.sh \
  --rir_exp exp/rir_train_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192 \
  "$@"

