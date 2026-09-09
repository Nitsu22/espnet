#!/usr/bin/env bash
# Generate WAVs and SCP manifests; does not run training or WSJ text preparation.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
python=${WHAMR_PYTHON:-/home/kslab/nitsu/.conda/envs/whamr/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
"${python}" "${recipe}/whamr_scripts/create_wham_from_scratch_rir.py" \
    --wsj0-root "${recipe}/data/wsj0/wsj0_wav" \
    --wham-noise-root /net/midgar/work2/nitsu/data/wsj/wham_noise \
    --output-dir "${recipe}/data_rir_plus" \
    --sample-rates 8k 16k --data-lengths min max --splits tr cv tt \
    "$@"
