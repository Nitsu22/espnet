#!/usr/bin/env bash
# Resources are selected by the controller; submit from the dedicated recipe.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qsub qacct >/dev/null
cd "${SGE_O_WORKDIR:?}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*) ;; *) exit 2 ;; esac
[[ ${JOB_ID:?} =~ ^[0-9]+$ ]]
case " $* " in
    *' --variant concat_glue '*) suffix=concat_glue ;;
    *) suffix=film_all ;;
esac
logs=exp/enh_train_tflocoformer_s_nf8k_recrir_ctf_${suffix}_seed0/.pipeline/job_${JOB_ID}
mkdir -p "${logs}"
task_tmp=$(mktemp -d "${PWD}/${logs}/tmp.XXXXXX")
export TMPDIR=${task_tmp} XDG_CACHE_HOME=${task_tmp}/cache
export NUMBA_CACHE_DIR=${task_tmp}/numba TRITON_CACHE_DIR=${task_tmp}/triton
export MPLCONFIGDIR=${task_tmp}/matplotlib PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    rm -rf -- "${task_tmp}"
    date -Is
    exit "${code}"
}
trap cleanup EXIT
date -Is
git rev-parse HEAD
case " $* " in *' --phase train '*|*' --phase infer '*) module load cuda/11.8.0 ;; esac
python local/nf8k_recrir_pipeline.py "$@"
