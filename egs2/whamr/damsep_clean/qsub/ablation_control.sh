#!/usr/bin/env bash
# Lightweight accounting check and continuation/scoring submission.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:20:00
#$ -N damsep_abl_control
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
# Compute nodes do not inherit the login node scheduler tools.
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${PWD}/exp/.job_tmp/ablation_${JOB_ID}"
mkdir -p -m 700 "${TMPDIR}"
trap 'rm -rf -- "${TMPDIR}"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for command in qacct qsub qrls; do command -v "${command}" >/dev/null; done
python local/ablation_pipeline.py control "$@"
