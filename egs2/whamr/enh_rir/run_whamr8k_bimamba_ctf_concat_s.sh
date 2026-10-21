#!/usr/bin/env bash
set -euo pipefail
# Keep all noise-free and FiLM experiments unchanged. BiMamba is fixed.
exec bash ./run_nf16k_bimamba_ctf_film_all_s.sh \
    --sample_rate 8000 --input_style baseline --dataset_condition whamr \
    --dumpdir dump_whamr_baseline_8k \
    --prepared_inputs exp/bimamba_whamr8k_baseline_inputs \
    --enh_config conf/tuning/train_enh_tflocoformer_s_whamr8k_bimamba_ctf_concat.yaml \
    --enh_exp exp/enh_train_tflocoformer_s_whamr8k_bimamba_ctf_concat_seed0 "$@"
