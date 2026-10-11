#!/usr/bin/env bash
# Resources and time limits are supplied by local/baseline_pipeline.py.
#$ -cwd
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_pipeline_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_pipeline_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
case " $* " in
    *" --phase gpu "*) module load cuda/11.8.0 ;;
esac
python local/baseline_pipeline.py "$@"
