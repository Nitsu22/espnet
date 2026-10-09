#!/usr/bin/env bash
# Submit from the dedicated rir_2spk checkout; optional argument: resume=true.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N bimamba_sweep_drr
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
recipe=${SGE_O_WORKDIR:?Submit from the dedicated recipe directory}
shared_recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
resume=${1:-false}
case "${resume}" in true|false) ;; *) exit 2 ;; esac
cd "${recipe}"
test -f run_pooled_bimamba_2spk_nf_16k_sweep_v2_drr.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${recipe}/../../..:${shared_recipe}/../damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_drr_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_drr_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bimamba_drr_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
logs="exp/pooled_bimamba_sweep_drr_jobs/job_${JOB_ID}"
mkdir -p "${logs}"
trap 'code=$?; printf "%s\n" "$code" > "${logs}/exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD | tee "${logs}/commit.txt"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.device_count() == 1, 'Exactly one visible GPU is required'
print(torch.cuda.get_device_name())
PY
python local/prepare_two_speaker_dump.py --output dump_nf_2spk_16k_min
# Header/manifest-based Stage 5 is lightweight; combine it here to avoid a
# separate minimum-charge job. Existing immutable audio is not regenerated.
bash run_pooled_bimamba_2spk_nf_16k_sweep_v2_drr.sh \
    --stage 5 --stop_stage 5 --ngpu 0 --nj 4 \
    --python "${CONDA_PREFIX}/bin/python" > "${logs}/stage5.log" 2>&1
cat "${logs}/stage5.log"
python local/check_two_speaker_training.py \
    --config conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2_drr.yaml \
    --data-dir dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k \
    --valid-data-dir dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k \
    --batch-size 4 --warmup-steps 3 --timed-steps 8 \
    --output "${logs}/profile.json" > "${logs}/profile.log" 2>&1
cat "${logs}/profile.log"
# The profile model is discarded. Training resets the config seed and starts
# from scratch unless explicitly continuing this experiment's checkpoint.
remaining_seconds=$((86400 - SECONDS - 120))
[[ ${remaining_seconds} -gt 300 ]]
set +e
timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
    bash run_pooled_bimamba_2spk_nf_16k_sweep_v2_drr.sh \
    --stage 6 --stop_stage 6 --ngpu 1 --nj 4 \
    --python "${CONDA_PREFIX}/bin/python" --resume "${resume}"
training_status=$?
set -e
if [[ ${training_status} -eq 124 && -s exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_drr/checkpoint.pth ]]; then
    printf '%s\n' 'Time limit reached; resume this checkpoint' > "${logs}/resume_ready"
fi
exit "${training_status}"
