#!/usr/bin/env bash
# Submit only after CPU preparation and the GPU validation have succeeded.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N tfs_oracle_ctf
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/oracle_ctf_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/oracle_ctf_mpl_${JOB_ID}"
resume=${1:-false}
python local/check_oracle_ctf_readiness.py
bash run_oracle_ctf_concat.sh --stage 6 --stop_stage 6 --ngpu 1 \
    --python "${CONDA_PREFIX}/bin/python" --resume "${resume}"
