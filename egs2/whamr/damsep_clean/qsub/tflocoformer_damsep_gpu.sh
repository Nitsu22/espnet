#!/usr/bin/env bash
# Submit via local/tflocoformer_damsep_pipeline.py from the recipe directory.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N tf_damsepation
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
export PATH="${PATH}:/usr/sbin:/sbin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${PWD}/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_tfdamsep_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_tfdamsep_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_tfdamsep_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
python local/tflocoformer_damsep_pipeline.py gpu "$@"
