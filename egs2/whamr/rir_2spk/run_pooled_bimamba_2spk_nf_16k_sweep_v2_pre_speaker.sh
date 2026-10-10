#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_2spk_nf_16k.sh" \
    --rir_model_type pooled_bimamba_sweep_v2_pit \
    --rir_config conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2_pre_speaker.yaml \
    --rir_tag train_pooled_bimamba_2spk_nf_16k_sweep_v2_pre_speaker "$@"
