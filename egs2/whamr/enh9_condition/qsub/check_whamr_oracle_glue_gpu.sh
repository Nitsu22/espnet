#!/usr/bin/env bash
# Reuse the verified WHAMR dump; only the delimiter setting changes.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:20:00
#$ -p -5
#$ -N oracle_glue_check
#$ -o qsub_logs
#$ -e qsub_logs
source qsub/oracle_environment.sh
module load cuda/11.8.0
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qsub qacct >/dev/null
python local/check_oracle_ctf_training.py \
    --config conf/tuning/train_enh_tflocoformer_s_whamr8k_oracle_ctf_concat_glue.yaml \
    --condition whamr --dump dump_oracle_ctf_whamr8k --device cuda \
    --output exp/oracle_ctf_whamr_glue_cuda_check.json
python local/oracle_ctf_pipeline.py --condition whamr --variant concat_glue --phase control
