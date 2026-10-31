#!/usr/bin/env bash
set -euo pipefail
exec bash ./run_nf16k_bimamba_ctf_film_all_s.sh \
    --sample_rate 8000 --input_style baseline \
    --enh_config conf/tuning/train_enh_tflocoformer_s_nf8k_recrir_ctf_film_all.yaml \
    --enh_exp exp/enh_train_tflocoformer_s_nf8k_recrir_ctf_film_all_seed0 \
    --normalize_output_wav true "$@"
