#!/usr/bin/env bash
# Lightweight accounting check and continuation/scoring submission.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:20:00
#$ -N tf_damsep_control
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
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qacct qsub qrls >/dev/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python local/tflocoformer_damsep_pipeline.py control "$@"
