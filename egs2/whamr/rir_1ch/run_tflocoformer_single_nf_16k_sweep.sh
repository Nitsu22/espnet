#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_single_nf_16k.sh" \
    --rir_model_type rec_rir_sweep \
    --rir_config conf/tuning/train_tflocoformer_single_nf_16k_sweep.yaml \
    --rir_tag train_tflocoformer_single_nf_16k_sweep "$@"
