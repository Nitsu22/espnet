#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=01:00:00
#$ -N tfs_nf16_prep
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python local/prepare_nf16k_baseline.py --source \
    /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_nf_2spk_16k_min
