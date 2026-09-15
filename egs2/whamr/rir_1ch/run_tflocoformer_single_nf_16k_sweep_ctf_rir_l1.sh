#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_tflocoformer_single_nf_16k_sweep.sh" \
    --rir_config conf/tuning/train_tflocoformer_single_nf_16k_sweep_ctf_rir_l1.yaml \
    --rir_tag train_tflocoformer_single_nf_16k_sweep_ctf_rir_l1_input_batch "$@"
