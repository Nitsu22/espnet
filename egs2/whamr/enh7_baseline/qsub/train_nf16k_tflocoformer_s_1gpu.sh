#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N tfs_nf16_b4
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_nf16_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_nf16_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
git rev-parse HEAD
expdir=exp/enh_train_tflocoformer_s_nf_16k_1gpu_batch4
python local/check_nf16k_training.py --condition nf_whamr --expdir "${expdir}"
bash run_nf16k_tflocoformer_s.sh --stage 6 --stop_stage 6 --ngpu 1
