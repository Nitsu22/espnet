#!/usr/bin/env bash
# Submit from the clean-only checkout's egs2/whamr/damsep_clean.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N damsep_pair_infer
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
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_eval_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_damsep_eval_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_damsep_eval_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
if (( $# == 0 )); then
    set -- /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/damsep_clean/exp/damsep_nf_8k \
        "${PWD}/exp/damsep_nf_8k_clean_only"
fi
printf 'Evaluation code commit: %s\n' "$(git rev-parse HEAD)"
for expdir in "$@"; do
    test -s "${expdir}/config.yaml"
    test -s "${expdir}/valid.loss.best.pth"
    printf 'Evaluating %s, model %s\n' "${expdir}" "$(readlink -f "${expdir}/valid.loss.best.pth")"
    bash run.sh --stage 7 --stop_stage 7 --ngpu 1 --expdir "${expdir}"
done
