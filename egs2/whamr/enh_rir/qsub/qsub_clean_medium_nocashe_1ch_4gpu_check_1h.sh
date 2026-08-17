#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=01:00:00
#$ -N m_base_chk
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

./run_clean_medium_nocashe_1ch.sh \
    --ngpu 4 \
    --stage 6 \
    --stop_stage 6 \
    --enh_args "--batch_size 8 --accum_grad 1 --seed 0"
