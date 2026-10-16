#!/usr/bin/env bash
# Probe weights are discarded; successful validation submits the full pipeline.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:20:00
#$ -p -5
#$ -N oracle_film_check
#$ -o qsub_logs
#$ -e qsub_logs
source qsub/oracle_environment.sh
module load cuda/11.8.0
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qsub qacct >/dev/null
python local/check_oracle_ctf_training.py \
    --config conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_film.yaml \
    --dump dump_oracle_ctf_nf8k --device cuda \
    --output exp/oracle_ctf_film_cuda_check.json
python local/oracle_ctf_pipeline.py --variant film --phase control
