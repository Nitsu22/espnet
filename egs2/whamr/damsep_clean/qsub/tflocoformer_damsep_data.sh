#!/usr/bin/env bash
# Shared noisy-input/statistics checks before either paid GPU experiment.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:10:00
#$ -N tfd_data_check
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
# Keep all runtime writes in this user's approved experiment storage.
export TMPDIR="${PWD}/exp/.runtime/tfdamsep_${JOB_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf -- "${TMPDIR}"' EXIT
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${TMPDIR}/cache"
export CUDA_CACHE_PATH="${TMPDIR}/cuda"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_tfdamsep_data_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_tfdamsep_data_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
python local/check_tflocoformer_damsep.py --variant clean_reverb --data-only \
    --report exp/tflocoformer_damsep_data_check.json
