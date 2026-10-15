#!/usr/bin/env bash
# Initial one-hour compatibility/training check; optional checkpoint-only resume.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -p -5
#$ -N tfs_bi8_concat
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
case ${PWD} in
    /gs/bs/tga-shinoda/nitsu/*) ;;
    *) echo 'Use an experiment checkout in the permitted personal directory' >&2; exit 2 ;;
esac
allocation_seconds=${1:-3600}
require_checkpoint=${2:-false}
[[ ${allocation_seconds} =~ ^[0-9]+$ ]] && (( allocation_seconds >= 600 ))
experiment=exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0
case ${require_checkpoint} in
    true) test -s "${experiment}/checkpoint.pth" ;;
    false)
        # A fresh experiment must not silently reuse unrelated checkpoints.
        test ! -e "${experiment}/checkpoint.pth" ;;
    *) echo 'require_checkpoint must be true or false' >&2; exit 2 ;;
esac
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
logs="${PWD}/exp/bimamba_nf8k_concat_jobs/job_${JOB_ID:?}"
mkdir -p "${logs}"
job_tmp=$(mktemp -d "${logs}/tmp.XXXXXX")
export TMPDIR="${job_tmp}" XDG_CACHE_HOME="${job_tmp}/cache"
export NUMBA_CACHE_DIR="${job_tmp}/numba" TRITON_CACHE_DIR="${job_tmp}/triton"
export MPLCONFIGDIR="${job_tmp}/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    case ${job_tmp} in
        "${logs}"/tmp.*) rm -rf -- "${job_tmp}" ;;
    esac
    date -Is
    exit "${code}"
}
trap cleanup EXIT
date -Is
hostname
git rev-parse HEAD | tee "${logs}/commit.txt"
# Read-only allocation diagnostics. Never change the scheduler's GPU mapping.
{
    printf 'shell_pid=%s parent_pid=%s\n' "$$" "${PPID}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
    printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES-<unset>}"
    sed -n '/^NSpid:/p' "/proc/$$/status"
} > "${logs}/process_environment.txt"
qstat -j "${JOB_ID}" > "${logs}/scheduler.txt" 2>&1 || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L > "${logs}/gpu_devices.txt" 2>&1 || true
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,mig.mode.current \
        --format=csv > "${logs}/gpu_before.txt" 2>&1 || true
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
        --format=csv > "${logs}/gpu_processes_before.txt" 2>&1 || true
fi
test -s exp/bimamba_nf8k_baseline_inputs/preparation.json
python local/check_bimamba_conditioning_training.py \
    --config conf/tuning/train_enh_tflocoformer_s_nf8k_bimamba_ctf_concat.yaml \
    --dump exp/bimamba_nf8k_baseline_inputs --input-style baseline \
    --output "${logs}/cuda_check.json" > "${logs}/cuda_check.log" 2>&1
cat "${logs}/cuda_check.log"
# Discard probe weights; the actual Stage 6 starts with the configured seed.
# Never reduce batch size or switch precision automatically after an OOM.
remaining_seconds=$((allocation_seconds - SECONDS - 120))
(( remaining_seconds > 300 ))
set +e
timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
    bash run_nf8k_bimamba_ctf_concat_s.sh --stage 6 --stop_stage 6 \
    --ngpu 1 --python "${CONDA_PREFIX}/bin/python"
training_status=$?
set -e
if [[ ${training_status} -eq 124 && -s ${experiment}/checkpoint.pth ]]; then
    printf '%s\n' 'Time limit reached; checkpoint available for resumption' > "${logs}/resume_ready"
    exit 0
fi
exit "${training_status}"
