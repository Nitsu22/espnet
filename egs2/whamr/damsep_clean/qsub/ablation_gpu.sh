#!/usr/bin/env bash
# Submit via local/ablation_pipeline.py from the recipe directory.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N damsep_ablation
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
# Compute nodes do not inherit the login node scheduler tools.
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${PWD}/exp/.job_tmp/ablation_${JOB_ID}"
mkdir -p -m 700 "${TMPDIR}"
trap 'rm -rf -- "${TMPDIR}"' EXIT
export PATH="${PATH}:/usr/sbin:/sbin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${PWD}/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_ablation_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_ablation_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_ablation_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
python local/ablation_pipeline.py gpu "$@"
