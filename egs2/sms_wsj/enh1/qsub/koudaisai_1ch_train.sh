#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -N koudai1ch_train
#$ -o qsub_logs/
#$ -e qsub_logs/
# Supply h_rt at submission after reviewing the measured preparation timings.
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/11.8.0
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
for split in train valid; do
    test -s "exp_tfgridnet_koudaisai_1ch/enh_stats_8k/${split}/speech_mix_shape"
done
bash run_tfgridnet_koudaisai_1ch.sh --stage 6 --stop_stage 6
