#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=01:00:00
#$ -N m_ctf_id
#$ -m abe
#$ -M daichi2ni2two@icloud.com
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
set -u
set -o pipefail

if __conda_setup="$(/gs/bs/tga-shinoda/nitsu/anaconda3/bin/conda shell.bash hook 2> /dev/null)"; then
    eval "${__conda_setup}"
else
    if [ -f /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh ]; then
        . /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
    else
        export PATH="/gs/bs/tga-shinoda/nitsu/anaconda3/bin:${PATH}"
    fi
fi
unset __conda_setup

module load cuda/11.8.0
conda activate tf-locoformer

export CUDA_VISIBLE_DEVICES=0,1,2,3

./run_rir_cmha_film_all_ctf_medium_nocashe_clean_1ch.sh \
    --prepare_ctf false \
    --gpu 0,1,2,3 \
    --ngpu 4 \
    --stage 6 \
    --stop_stage 6 \
    --enh_exp exp/rir_cmha_film_all_ctf_clean_1ch/enh_train_enh_tflocoformer_medium_nocashe_rir_cmha_film_all_ctf_identity_init_clean_1ch_seed0 \
    --enh_args "--batch_size 8 --accum_grad 1 --seed 0"
