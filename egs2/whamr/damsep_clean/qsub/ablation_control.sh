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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python local/ablation_pipeline.py control "$@"
