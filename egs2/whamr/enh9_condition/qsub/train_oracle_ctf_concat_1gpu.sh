#!/usr/bin/env bash
# Submit only after CPU preparation and the GPU validation have succeeded.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N tfs_oracle_ctf
#$ -o qsub_logs
#$ -e qsub_logs
source qsub/oracle_environment.sh
module load cuda/11.8.0
resume=${1:-false}
variant=${2:-concat}
condition=${3:-nf_whamr}
python local/check_oracle_ctf_readiness.py --variant "${variant}" --condition "${condition}"
bash run_oracle_ctf_concat.sh --stage 6 --stop_stage 6 --ngpu 1 \
    --python "${CONDA_PREFIX}/bin/python" --resume "${resume}" --variant "${variant}" --condition "${condition}"
