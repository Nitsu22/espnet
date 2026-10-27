#!/usr/bin/env bash
# Submit from enh_ctf_dereverb with -g tga-shinoda (charged validation).
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:20:00
#$ -N ctf31_check
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
. qsub/environment.sh
paired_dump=${1:?Pass the absolute path of the verified paired dump}
python local/check_paired_dump.py "${paired_dump}"
python local/check_cuda.py --config conf/train_ctf_joint.yaml \
    --paired_dump "${paired_dump}" --steps 4 \
    --output "exp/cuda_checks/check_${JOB_ID}.json"
