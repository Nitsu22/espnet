#!/usr/bin/env bash
# Submit from enh7_baseline; pass whamr or nf_whamr as the script argument.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N tfs_baseline_b4
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
condition=${1:?Pass whamr or nf_whamr}
case ${condition} in
    whamr) condition_tag=whamr ;;
    nf_whamr) condition_tag=nf ;;
    *) echo "Unknown condition: ${condition}" >&2; exit 2 ;;
esac
expdir=exp/enh_train_tflocoformer_s_${condition_tag}_8k_1gpu_batch4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_tfs_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_tfs_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
python local/check_baseline_inputs.py --condition "${condition}" --expdir "${expdir}"
bash run_tflocoformer_s.sh --condition "${condition}" \
    --stage 6 --stop_stage 6 --ngpu 1 --batch_size 4 --accum_grad 1 \
    --valid_batch_size 1 --enh_exp "${expdir}"
