#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
bash "${recipe}/run_rec_rir_single_nf_16k.sh" \
    --rir_model_type rec_rir_sweep \
    --rir_config conf/tuning/train_rec_rir_single_sweep_16k.yaml \
    --rir_tag train_rec_rir_single_sweep_16k "$@"
