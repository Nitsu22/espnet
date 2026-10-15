#!/usr/bin/env bash
# Run from an isolated, committed checkout; workspace holds immutable inputs.
set -euo pipefail
workspace=${1:?workspace root required}
output=${2:?new experiment output required}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python=${PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
export TMPDIR=/tmp/nitsu/damsep-ls8k-20261015
mkdir -p "${TMPDIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
finish() {
    status=$?
    printf '%s\n' "${status}" > "${output}/exit_status"
    date '+%Y-%m-%d %H:%M:%S' > "${output}/finished_at"
    rmdir "${TMPDIR}"
    exit "${status}"
}
trap finish EXIT
cd "${repo}"
date '+%Y-%m-%d %H:%M:%S' > "${output}/started_at"
hostname > "${output}/host"
git rev-parse HEAD > "${output}/git_commit"
"${python}" -B egs2/whamr/rir_2spk/local/evaluate_damsep_bimamba_ls8k.py \
    --workspace "${workspace}" --output "${output}" --device cuda:0
