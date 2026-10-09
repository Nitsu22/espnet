#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -p -5
#$ -N bimamba8_inputs
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
date -Is
git rev-parse HEAD
python local/prepare_bimamba_baseline8k_inputs.py \
    --dump dump_nf_baseline_8k --output exp/bimamba_nf8k_baseline_inputs --workers 4
