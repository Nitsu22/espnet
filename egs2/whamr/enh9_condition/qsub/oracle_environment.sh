# Shared setup for Oracle compute jobs. Source from the recipe directory.
if [[ ${ORACLE_TASK_ENV_READY:-0} == 1 ]]; then return 0; fi
set -eo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
if [[ -e /tmp/nitsu ]]; then
    [[ -d /tmp/nitsu && ! -L /tmp/nitsu && -O /tmp/nitsu ]] || exit 1
else
    mkdir -m 700 /tmp/nitsu
fi
oracle_task_tmp=$(mktemp -d "/tmp/nitsu/oracle_ctf_${JOB_ID:?}_XXXXXX")
trap 'rm -rf -- "${oracle_task_tmp}"' EXIT
export TMPDIR="${oracle_task_tmp}"
export NUMBA_CACHE_DIR="${oracle_task_tmp}/numba"
export MPLCONFIGDIR="${oracle_task_tmp}/matplotlib"
export XDG_CACHE_HOME="${oracle_task_tmp}/cache"
export PYTHONDONTWRITEBYTECODE=1
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export ORACLE_TASK_ENV_READY=1
