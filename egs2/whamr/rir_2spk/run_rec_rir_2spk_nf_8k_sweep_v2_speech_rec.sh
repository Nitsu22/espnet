#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh" \
    --rir_model_type rec_rir_sweep_v2_speech_pit \
    --rir_config conf/tuning/train_rec_rir_2spk_nf_8k_sweep_v2_speech_rec.yaml \
    --rir_tag train_rec_rir_2spk_nf_8k_sweep_v2_speech_rec "$@"
