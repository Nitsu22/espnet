#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_pooled_bimamba_2spk_nf_8k_sweep_v2.sh" \
    --rir_config conf/tuning/train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep.yaml \
    --rir_tag train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep "$@"
