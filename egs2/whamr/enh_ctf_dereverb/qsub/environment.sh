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
ctf_mamba_deps=${CTF_MAMBA_DEPS:-/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310}
if [[ ! -r ${ctf_mamba_deps}/mamba_ssm/__init__.py ]]; then
    echo "Missing isolated Mamba dependencies: ${ctf_mamba_deps}" >&2
    exit 2
fi
export PYTHONPATH="$(cd ../../.. && pwd):${ctf_mamba_deps}${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
printf 'Mamba dependencies: %s\n' "${ctf_mamba_deps}"
printf 'Node: %s\n' "$(hostname)"
date '+Start: %Y-%m-%d %H:%M:%S %Z'
