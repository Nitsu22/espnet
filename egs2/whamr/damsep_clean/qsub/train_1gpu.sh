#!/usr/bin/env bash
# Submit from egs2/whamr/damsep_clean with qsub -g tga-shinoda.
# Resume the same experiment by submitting this script again after it ends.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N damsep_nf_1gpu
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
export PATH="${PATH}:/usr/sbin:/sbin"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${PWD}/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_train_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_train_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_damsep_train_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
# Verify the Stage 5 sources and full-length validation CUDA path in this
# charged allocation, then start/resume training with effective batch one.
python local/check_training_inputs.py
bash run.sh --stage 6 --stop_stage 6 --ngpu 1 --accum_grad 1 --resume true
