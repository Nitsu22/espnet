#!/usr/bin/env bash
# Sourced by this recipe's qsub jobs; do not submit this helper directly.
set -eo pipefail
if [[ -e /tmp/nitsu && ! -O /tmp/nitsu ]]; then
    echo "/tmp/nitsu is not owned by this user" >&2
    exit 2
fi
mkdir -p /tmp/nitsu
ctf_task_tmp=$(mktemp -d "/tmp/nitsu/ctf-tsubame.${JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "${ctf_task_tmp}"' EXIT
trap 'exit 143' TERM
export TMPDIR=${ctf_task_tmp}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="${PWD}/.cache"
export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -u
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
printf 'Node: %s\n' "$(hostname)"
date '+Start: %Y-%m-%d %H:%M:%S %Z'
