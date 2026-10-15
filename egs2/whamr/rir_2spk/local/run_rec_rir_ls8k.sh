#!/usr/bin/env bash
set -euo pipefail
workspace=${1:?workspace root required}
output=${2:?new output directory required}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python=${PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
export TMPDIR=/tmp/nitsu/rec-rir-ls8k-20261015
mkdir -p "${TMPDIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
finish() {
    status=$?
    printf '%s\n' "${status}" > "${output}.exit_status"
    date '+%Y-%m-%d %H:%M:%S' > "${output}.finished_at"
    rmdir "${TMPDIR}"
    exit "${status}"
}
trap finish EXIT
cd "${repo}"
date '+%Y-%m-%d %H:%M:%S' > "${output}.started_at"
git rev-parse HEAD > "${output}.git_commit"
hostname > "${output}.host"
"${python}" -B egs2/whamr/rir_2spk/local/evaluate_rec_rir_ls8k.py \
    --workspace "${workspace}" --output "${output}" \
    --previous "${workspace}/egs2/whamr/rir_2spk/exp/damsep_bimamba_ls8k_20261015"
