#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
exec bash "${recipe}/run_rec_rir_single_nf_16k.sh" \
    --rir_args "--valid_batch_size 1" \
    --batch_by_input_only true --speech_fold_length 160000 \
    --rir_model_type rec_rir_sweep \
    --rir_config conf/tuning/train_tflocoformer_single_nf_16k_sweep.yaml \
    --rir_tag train_tflocoformer_single_nf_16k_sweep_input_batch "$@"
