#!/usr/bin/env bash
# Compatibility check only. Submit without -g/newgrp; at most one trial running.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:03:00
#$ -N damsep_cuda_check
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${PWD}/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_cuda_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_triton_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
python local/check_cuda.py
