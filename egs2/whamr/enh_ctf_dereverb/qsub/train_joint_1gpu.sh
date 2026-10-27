#!/usr/bin/env bash
# Submit from enh_ctf_dereverb after the charged CUDA check succeeds.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N ctf31_joint
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
. qsub/environment.sh
paired_dump=${1:?Pass the absolute path of the verified paired dump}
input_check=${2:?Pass the successful CUDA check JSON for this commit/config}
python local/check_cuda.py --config conf/train_ctf_joint.yaml \
    --paired_dump "${paired_dump}" --verify_report "${input_check}"
expdir=exp/enh_train_ctf_joint_whamr_8k_1gpu_batch4
resume=false
if [[ -f ${expdir}/checkpoint.pth ]]; then resume=true; fi
bash run.sh --phase joint --paired_dump "${paired_dump}" \
    --ngpu 1 --resume "${resume}" --enh_exp "${expdir}"
