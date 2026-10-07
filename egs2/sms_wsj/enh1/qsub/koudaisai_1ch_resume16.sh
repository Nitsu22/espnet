#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -p -4
#$ -N koudai1ch_b16
#$ -o qsub_logs/
#$ -e qsub_logs/
set -euo pipefail
exp_dir=exp_tfgridnet_koudaisai_1ch/tfgridnet_2block_1ch
test -s "${exp_dir}/checkpoint.pth"
# Preserve the preceding batch-4 run before ESPnet rewrites config and logs.
archive_dir="${exp_dir}/before_batch16_${JOB_ID}"
mkdir -p "$archive_dir"
cp -p "${exp_dir}/checkpoint.pth" "${exp_dir}/config.yaml" "${exp_dir}/train.log" "$archive_dir/"
export KOUDAISAI_BATCH_SIZE=16
exec bash qsub/koudaisai_1ch_train.sh
