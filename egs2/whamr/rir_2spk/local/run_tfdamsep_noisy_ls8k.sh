#!/usr/bin/env bash
set -euo pipefail
workspace=${1:?workspace root required}
source=${2:?source manifest and CTF directory required}
output=${3:?new output directory required}
limit=${4:-0}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python=${PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
[[ ! -e ${output} ]] || { echo "Output already exists" >&2; exit 2; }
export TMPDIR=/tmp/nitsu/tfdamsep-noisy-ls8k-20261028-$$
mkdir -p "${TMPDIR}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=${TMPDIR}/cache
export CUDA_CACHE_PATH=${TMPDIR}/cuda
export MPLCONFIGDIR=${TMPDIR}/matplotlib
export NUMBA_CACHE_DIR=${TMPDIR}/numba
export TRITON_CACHE_DIR=${TMPDIR}/triton
export TORCH_EXTENSIONS_DIR=${TMPDIR}/torch_extensions
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
finish() {
    status=$?
    printf '%s\n' "${status}" > "${output}.exit_status"
    date '+%Y-%m-%d %H:%M:%S' > "${output}.finished_at"
    rm -rf -- "${TMPDIR}"
    exit "${status}"
}
trap finish EXIT
cd "${repo}"
date '+%Y-%m-%d %H:%M:%S' > "${output}.started_at"
hostname > "${output}.host"
git rev-parse HEAD > "${output}.git_commit"
"${python}" -B egs2/whamr/rir_2spk/local/evaluate_tfdamsep_noisy_ls8k.py \
    --workspace "${workspace}" --source "${source}" --output "${output}" --limit "${limit}"
