#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_single_nf_16k.sh" \
    --rir_config conf/tuning/train_rec_rir_single_nf_16k_ctf_only.yaml \
    --rir_tag train_rec_rir_single_nf_16k_ctf_only "$@"
