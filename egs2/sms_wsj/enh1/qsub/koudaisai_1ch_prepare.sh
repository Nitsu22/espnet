#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=00:30:00
#$ -N koudai1ch_prep
#$ -o qsub_logs/
#$ -e qsub_logs/
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/11.8.0
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p exp_tfgridnet_koudaisai_1ch
trap 'code=$?; printf "%s exit=%s\n" "$(date -Is)" "$code" > exp_tfgridnet_koudaisai_1ch/prepare.status' EXIT
# Statistics need no GPUs. Four jobs with one data worker each fit cpu_16.
bash run_tfgridnet_koudaisai_1ch.sh --stage 5 --stop_stage 5 --ngpu 0 \
    --nj 4 --enh_args "--num_workers 1" \
    > exp_tfgridnet_koudaisai_1ch/stats.log 2>&1
