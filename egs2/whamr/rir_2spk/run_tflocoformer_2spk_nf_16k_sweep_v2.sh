#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_2spk_nf_16k.sh" \
    --rir_model_type tflocoformer_sweep_v2_pit \
    --rir_config conf/tuning/train_tflocoformer_2spk_nf_16k_sweep_v2.yaml \
    --rir_tag train_tflocoformer_2spk_nf_16k_sweep_v2 "$@"
