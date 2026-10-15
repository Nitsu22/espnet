#!/usr/bin/env bash
set -euo pipefail
# Explicit defaults keep all existing FiLM experiment paths untouched.
exec bash ./run_nf8k_bimamba_ctf_film_all_s.sh \
    --enh_config conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_concat.yaml \
    --enh_exp exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0 "$@"
