#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE'; exit 2; }
gpu_index=$1
result_recipe=$(realpath "$2")
[[ ${gpu_index} =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "${recipe}"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
config=conf/tuning/train_rec_rir_2spk_noisy_8k_all_loss.yaml
run_dir=${result_recipe}/exp/rec_rir_noisy8k_training_20261031
model_dir=${result_recipe}/exp/rir_train_rec_rir_2spk_noisy_8k_all_loss
stats_dir=${result_recipe}/exp/rir_stats_rec_rir_2spk_noisy_8k_all_loss
dump=${result_recipe}/dump_noisy_2spk_8k_min
nf_dump=${result_recipe}/dump_nf_2spk_8k_min
for path in "${run_dir}" "${model_dir}" "${stats_dir}"; do
    [[ ! -e ${path} ]] || { echo "Use a fresh output: ${path}"; exit 1; }
done
mkdir -p "${run_dir}/.tmp"
export CUDA_VISIBLE_DEVICES=${gpu_index} TZ=Asia/Tokyo
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR=${run_dir}/.tmp XDG_CACHE_HOME=${run_dir}/.tmp/cache
export NUMBA_CACHE_DIR=${TMPDIR}/numba TRITON_CACHE_DIR=${TMPDIR}/triton
export TORCH_EXTENSIONS_DIR=${TMPDIR}/torch_extensions MPLCONFIGDIR=${TMPDIR}/matplotlib
export CUDA_CACHE_PATH=${TMPDIR}/cuda TORCH_HOME=${TMPDIR}/torch
export XDG_CONFIG_HOME=${TMPDIR}/config TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/torchinductor
phase=preparation
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "${status}" > "${run_dir}/exit_status"
    printf '%s\n' "${phase}" > "${run_dir}/final_phase"
    date -Is > "${run_dir}/finished_at"
    rm -rf -- "${run_dir}/.tmp"
    exit "${status}"
}
trap finish EXIT
git rev-parse HEAD > "${run_dir}/git_commit"
git -C "${code_root}" diff --quiet HEAD -- espnet2 egs2/whamr/rir_2spk
date -Is > "${run_dir}/started_at"
hostname > "${run_dir}/host"
common=(--config "${config}" --dump "${dump}" --nf-dump "${nf_dump}"
        --stats "${stats_dir}" --run "${run_dir}" --model "${model_dir}")
"${python}" -B local/recrir_noisy8k.py prepare "${common[@]}"
# Check the chosen host GPU again immediately before loading CUDA weights.
"${python}" -B - "${gpu_index}" "${run_dir}" <<'PY'
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

gpu, run = int(sys.argv[1]), Path(sys.argv[2])
items = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in items if int(row[0]) == gpu)
processes = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in processes):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
(run / 'gpu.json').write_text(json.dumps(dict(index=gpu, uuid=selected[1], model=selected[2]), indent=2) + '\n')
PY
phase=gpu_verification
"${python}" -B local/recrir_noisy8k.py probe "${common[@]}"
phase=training
bash ./rir.sh --stage 6 --stop_stage 6 --skip_data_prep true --resume false \
    --python "${python}" --fs 8k --ngpu 1 --num_nodes 1 --nj 4 \
    --dumpdir "${dump}" --train_set tr_rir_2spk_noisy_min_8k \
    --valid_set cv_rir_2spk_noisy_min_8k --test_sets tt_rir_2spk_noisy_min_8k \
    --rir_model_type rec_rir_pit --rir_config "${config}" --rir_tag rec_rir_2spk_noisy_8k_all_loss \
    --rir_exp "${model_dir}" --rir_stats_dir "${stats_dir}" \
    --batch_by_input_only true --speech_fold_length 200000 \
    --rir_args '--batch_size 4 --valid_batch_size 1 --accum_grad 1'
phase=finalization
"${python}" -B local/recrir_noisy8k.py finalize "${common[@]}"
phase=evaluation
"${python}" -B local/recrir_noisy8k.py evaluate "${common[@]}"
phase=complete
