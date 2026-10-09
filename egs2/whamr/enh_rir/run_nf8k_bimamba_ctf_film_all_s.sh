#!/usr/bin/env bash
set -euo pipefail
# Share the tested training/inference path, selecting the real 8-kHz
# checkpoint and enh7_baseline's exact NF-WHAMR observation/teacher WAVs.
exec bash ./run_nf16k_bimamba_ctf_film_all_s.sh \
    --sample_rate 8000 --input_style baseline "$@"
