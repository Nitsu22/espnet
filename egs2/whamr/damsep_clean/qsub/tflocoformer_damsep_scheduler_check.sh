#!/usr/bin/env bash
# Check scheduler access on a compute node before enabling continuations.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:05:00
#$ -N tfd_scheduler_check
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -euo pipefail
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qacct qsub qrls
qacct -j "${1:?Pass a completed GPU job ID}"
