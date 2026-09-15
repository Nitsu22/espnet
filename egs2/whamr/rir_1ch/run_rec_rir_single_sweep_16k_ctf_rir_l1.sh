#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_single_sweep_16k_input_batch.sh" \
    --rir_config conf/tuning/train_rec_rir_single_sweep_16k_ctf_rir_l1.yaml \
    --rir_tag train_rec_rir_single_sweep_16k_ctf_rir_l1_input_batch "$@"
