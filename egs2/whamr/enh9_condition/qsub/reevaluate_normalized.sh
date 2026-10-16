#!/usr/bin/env bash
# Resource and dependency are selected explicitly at submission.
#$ -cwd
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
set -eo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
phase=${1:?Use infer or score}
case "${phase}" in infer|score) ;; *) exit 2 ;; esac
source_exp=exp/enh_train_tflocoformer_s_nf8k_oracle_ctf_concat_seed0
evaluation_exp=${source_exp}/evaluation_normalized_20261016
if [[ -e /tmp/nitsu ]]; then
    [[ -d /tmp/nitsu && ! -L /tmp/nitsu && -O /tmp/nitsu ]] || exit 1
else
    mkdir -m 700 /tmp/nitsu
fi
task_tmp=$(mktemp -d "/tmp/nitsu/oracle_normalized_${JOB_ID:?}_XXXXXX")
trap 'rm -rf -- "${task_tmp}"' EXIT
export TMPDIR="${task_tmp}"
export NUMBA_CACHE_DIR="${task_tmp}/numba"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export XDG_CACHE_HOME="${task_tmp}/cache"
export PYTHONDONTWRITEBYTECODE=1
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ${phase} == infer ]]; then
    module load cuda/11.8.0
    for marker in inference.done.json score.done.json; do
        [[ ! -e ${evaluation_exp}/.pipeline/${marker} ]] || {
            echo "Use a new evaluation directory after a completed inference" >&2
            exit 1
        }
    done
fi
python local/prepare_normalized_evaluation.py --source "${source_exp}" --output "${evaluation_exp}"
python local/oracle_ctf_pipeline.py --phase "${phase}" --exp "${evaluation_exp}"
python local/prepare_normalized_evaluation.py --source "${source_exp}" --output "${evaluation_exp}"
