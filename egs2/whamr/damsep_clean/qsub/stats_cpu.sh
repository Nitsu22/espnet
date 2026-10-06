#!/usr/bin/env bash
# Submit from egs2/whamr/damsep_clean with qsub -g tga-shinoda.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -N damsep_stats
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_stats_numba_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
bash run.sh --stage 5 --stop_stage 5 --ngpu 0 --nj 4
