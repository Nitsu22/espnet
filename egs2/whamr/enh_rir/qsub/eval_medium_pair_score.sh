#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=02:00:00
#$ -N m_pair_score
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -euo pipefail
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
./run_clean_medium_nocashe_1ch.sh --stage 8 --stop_stage 8 --ngpu 0 --gpu_inference false --inference_nj 8 --inference_model valid.loss.best.pth
./run_rir_cmha_film_all_ctf_medium_nocashe_clean_1ch.sh --prepare_ctf false --stage 8 --stop_stage 8 --ngpu 0 --gpu_inference false --inference_nj 8 --inference_model valid.loss.best.pth --enh_exp exp/rir_cmha_film_all_ctf_clean_1ch/enh_train_enh_tflocoformer_medium_nocashe_rir_cmha_film_all_ctf_identity_init_clean_1ch_seed0
