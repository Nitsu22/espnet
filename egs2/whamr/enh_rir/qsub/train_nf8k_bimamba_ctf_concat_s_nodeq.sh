#!/usr/bin/env bash
# Same one-GPU training/checkpoint path, with node_q as the ONLY resource type.
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N tfs_bi8_nodeq
#$ -o qsub_logs
#$ -e qsub_logs
set -euo pipefail
# These nested #$ comments are not parsed by qsub: the existing script runs
# under bash, retaining batch 4 / accum_grad 1 / FP32 and the same experiment.
exec bash qsub/train_nf8k_bimamba_ctf_concat_s_1gpu.sh "${1:-86400}" "${2:-true}"
