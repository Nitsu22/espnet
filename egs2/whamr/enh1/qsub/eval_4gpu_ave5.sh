#!/usr/bin/env bash
# Resource and h_rt are supplied by qsub: GPU inference, then held CPU scoring.
#$ -cwd
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -eo pipefail
if [[ -e /tmp/nitsu ]]; then
    [[ -d /tmp/nitsu && ! -L /tmp/nitsu && -O /tmp/nitsu ]] || exit 1
else
    mkdir -m 700 /tmp/nitsu
fi
task_tmp=$(mktemp -d /tmp/nitsu/whamr_ave5.XXXXXX)
trap 'rm -rf -- "${task_tmp}"' EXIT
export TMPDIR="${task_tmp}"
export XDG_CACHE_HOME="${TMPDIR}/cache" MPLCONFIGDIR="${TMPDIR}/mpl"
export NUMBA_CACHE_DIR="${TMPDIR}/numba" CUDA_CACHE_PATH="${TMPDIR}/cuda"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${CUDA_CACHE_PATH}"
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
if [[ " $* " == *" --phase infer "* ]]; then
    module load cuda/11.8.0
fi
python -u local/eval_4gpu_ave5.py "$@"
