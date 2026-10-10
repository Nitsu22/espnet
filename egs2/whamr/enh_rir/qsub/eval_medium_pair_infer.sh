#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=04:00:00
#$ -N m_pair_infer
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -euo pipefail
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -s exp/enh_train_enh_tflocoformer_medium_nocashe_clean_1ch/valid.loss.best.pth
test -s exp/rir_cmha_film_all_ctf_clean_1ch/enh_train_enh_tflocoformer_medium_nocashe_rir_cmha_film_all_ctf_identity_init_clean_1ch_seed0/valid.loss.best.pth
./run_clean_medium_nocashe_1ch.sh --stage 7 --stop_stage 7 --ngpu 1 --inference_nj 1 --inference_model valid.loss.best.pth
./run_rir_cmha_film_all_ctf_medium_nocashe_clean_1ch.sh --prepare_ctf false --gpu "${CUDA_VISIBLE_DEVICES:-0}" --stage 7 --stop_stage 7 --ngpu 1 --inference_nj 1 --inference_model valid.loss.best.pth --enh_exp exp/rir_cmha_film_all_ctf_clean_1ch/enh_train_enh_tflocoformer_medium_nocashe_rir_cmha_film_all_ctf_identity_init_clean_1ch_seed0
