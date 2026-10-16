#!/usr/bin/env bash
set -euo pipefail
workspace=${1:?workspace required}
output=${2:?output directory required}
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python=${PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
export TMPDIR=/tmp/nitsu/rec-rir-speech-20261016
mkdir -p "${TMPDIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
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
"${python}" -B egs2/whamr/rir_2spk/local/evaluate_rec_rir_speech.py \
    --workspace "${workspace}" --output "${output}" --workers 8
