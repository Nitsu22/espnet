#!/usr/bin/env bash
# Default initial one-hour run. For continuation, override h_rt at submission
# and pass the allocation length in seconds followed by require_checkpoint=true.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -p -5
#$ -N tfs_bimamba8
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
allocation_seconds=${1:-3600}
require_checkpoint=${2:-false}
[[ ${allocation_seconds} =~ ^[0-9]+$ ]] && (( allocation_seconds >= 600 ))
case ${require_checkpoint} in
    true) test -s exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0/checkpoint.pth ;;
    false) ;;
    *) echo 'require_checkpoint must be true or false' >&2; exit 2 ;;
esac
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bimamba8_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/bimamba8_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bimamba8_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
logs="exp/bimamba_nf8k_jobs/job_${JOB_ID}"
mkdir -p "${logs}"
trap 'code=$?; printf "%s\n" "$code" > "${logs}/exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD | tee "${logs}/commit.txt"
# A hold dependency only guarantees completion, not successful preparation.
test -s exp/bimamba_nf8k_baseline_inputs/preparation.json
python local/check_bimamba_conditioning_training.py \
    --config conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_film_all.yaml \
    --dump exp/bimamba_nf8k_baseline_inputs --input-style baseline \
    --output "${logs}/cuda_check.json" > "${logs}/cuda_check.log" 2>&1
cat "${logs}/cuda_check.log"
# Probe weights are discarded; Stage 6 initializes the configured seed anew.
remaining_seconds=$((allocation_seconds - SECONDS - 120))
[[ ${remaining_seconds} -gt 300 ]]
set +e
timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
    bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 6 --stop_stage 6 \
    --ngpu 1 --python "${CONDA_PREFIX}/bin/python"
training_status=$?
set -e
if [[ ${training_status} -eq 124 && -s exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0/checkpoint.pth ]]; then
    printf '%s\n' 'Time limit reached; checkpoint available for resumption' > "${logs}/resume_ready"
fi
exit "${training_status}"
