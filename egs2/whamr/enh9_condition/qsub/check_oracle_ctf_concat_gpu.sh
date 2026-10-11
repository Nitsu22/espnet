#!/usr/bin/env bash
# Paid GPU validation only; probe weights are discarded.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:15:00
#$ -p -5
#$ -N oracle_ctf_check
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/oracle_ctf_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/oracle_ctf_mpl_${JOB_ID}"
python local/check_oracle_ctf_training.py \
    --config conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml \
    --dump dump_oracle_ctf_nf8k --device cuda \
    --output exp/oracle_ctf_cuda_check.json
