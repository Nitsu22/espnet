#!/usr/bin/env bash
set -euo pipefail

# Default matches the requested 1-GPU batch-4 baseline. Override --enh_exp
# explicitly when changing batch/GPU settings or running another seed.
exec bash ./run_tflocoformer_s.sh \
    --condition whamr \
    --enh_config conf/tuning/train_enh_tflocoformer_bimamba_s_whamr_8k.yaml \
    --enh_exp exp/enh_train_tflocoformer_bimamba_s_whamr_8k_1gpu_batch4 \
    "$@"
