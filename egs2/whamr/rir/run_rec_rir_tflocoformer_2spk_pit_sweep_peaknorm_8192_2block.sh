#!/usr/bin/env bash

set -e
set -u
set -o pipefail

./run_rec_rir_tflocoformer_2spk_pit_clean_8192.sh \
  --rir_config conf/tuning/train_rec_rir_tflocoformer_2spk_pit_sweep_peaknorm_8192_2block.yaml \
  --rir_tag train_rec_rir_tflocoformer_2spk_pit_sweep_peaknorm_8192_2block \
  --rir_exp exp/rir_train_rec_rir_tflocoformer_2spk_pit_sweep_peaknorm_8192_2block \
  --rir_stats_dir exp/rir_stats_8k_rec_rir_pit_sweep \
  --use_sweep_target true \
  "$@"
