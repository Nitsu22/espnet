#!/usr/bin/env bash
# Run in a detached tmux session on an idle lab GPU; no dependency installation.
set -euo pipefail
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "${recipe}"
storage=${1:?Absolute personal recipe path for data and results}
gpu_index=${2:?Host GPU index}
launch_name=${3:?Unique launch directory basename}
case ${storage} in /net/midgar/work2/nitsu/*|/gs/bs/tga-shinoda/nitsu/*) ;; *) exit 2 ;; esac
[[ ${gpu_index} =~ ^[0-9]+$ && ${launch_name} =~ ^[a-zA-Z0-9_-]+$ ]]
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
logs="${storage}/exp/${launch_name}"
mkdir "${logs}"
job_tmp=$(mktemp -d "${logs}/tmp.XXXXXX")
export TMPDIR="${job_tmp}" XDG_CACHE_HOME="${job_tmp}/cache"
export NUMBA_CACHE_DIR="${job_tmp}/numba" TRITON_CACHE_DIR="${job_tmp}/triton"
export MPLCONFIGDIR="${job_tmp}/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${recipe}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${python}"):${PATH}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    date -Is > "${logs}/finished_at"
    case ${job_tmp} in "${logs}"/tmp.*) rm -rf -- "${job_tmp}" ;; esac
    exit "${code}"
}
trap cleanup EXIT
exec > "${logs}/launch.log" 2>&1
date -Is > "${logs}/started_at"
hostname > "${logs}/host"
git rev-parse HEAD > "${logs}/commit"
dump="${storage}/dump_noisy_2spk_8k_min"
experiment="${storage}/exp/rir_train_pooled_bimamba_2spk_noisy_8k_sweep_v2"
test ! -e "${experiment}/checkpoint.pth"
test -s "${dump}/preparation.json"
test -s "${storage}/exp/noisy_bimamba_8k_data_check.json"
# Check only the selected GPU and never take over an existing process.
nvidia-smi -i "${gpu_index}" --query-gpu=uuid,name,memory.used --format=csv \
    > "${logs}/gpu_before.csv"
active=$(nvidia-smi -i "${gpu_index}" --query-compute-apps=pid --format=csv,noheader)
[[ -z ${active} ]]
used=$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 256 ))
export CUDA_VISIBLE_DEVICES="${gpu_index}"
"${python}" local/check_two_speaker_training.py \
    --config conf/tuning/train_pooled_bimamba_2spk_noisy_8k_sweep_v2.yaml \
    --data-dir "${dump}/raw/tr_rir_2spk_noisy_min_8k" \
    --valid-data-dir "${dump}/raw/cv_rir_2spk_noisy_min_8k" \
    --noise --batch-size 4 --warmup-steps 1 --timed-steps 2 \
    --output "${logs}/gpu_check.json"
# Probe updates are discarded; training starts independently from seed zero.
printf '%s\n' training > "${logs}/phase"
bash run_pooled_bimamba_2spk_noisy_8k_sweep_v2.sh --stage 6 --stop_stage 6 \
    --resume false --python "${python}" --ngpu 1 --dumpdir "${dump}" \
    --rir_exp "${experiment}" \
    --rir_stats_dir "${storage}/exp/rir_stats_train_pooled_bimamba_2spk_noisy_8k_sweep_v2"
printf '%s\n' training_complete > "${logs}/phase"
