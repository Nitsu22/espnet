#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root=$(cd ../../.. && pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR=/tmp/nitsu_tfgridnet_numba
export OMP_NUM_THREADS=1
python_bin=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
run_dir=exp_tfgridnet_sms_wsj_1ch_paper
mkdir -p "${run_dir}"
trap 'code=$?; echo "$(date -Is) exit=${code}" > "${run_dir}/run.status"' EXIT
echo "$(date -Is) checking GPUs" > "${run_dir}/run.status"
check_gpus() {
"${python_bin}" - <<'PY'
import subprocess
uuids = set(subprocess.check_output([
    'nvidia-smi', '-i', '0,1,2,3', '--query-gpu=uuid', '--format=csv,noheader'
], text=True).splitlines())
processes = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'
], text=True).splitlines()
assert not any(line.split(',')[0] in uuids for line in processes), 'Requested GPU is in use'
PY
}
check_gpus
echo "$(date -Is) 4-GPU smoke test" > "${run_dir}/run.status"
"${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  local/check_tfgridnet_sms_wsj_1ch_paper.py > "${run_dir}/smoke_gpu.log" 2>&1
echo "$(date -Is) Stage 5" > "${run_dir}/run.status"
bash run_tfgridnet_sms_wsj_1ch_paper.sh --stop_stage 5 > "${run_dir}/run.log" 2>&1
# Statistics may take time; do not assume the GPUs stayed free meanwhile.
check_gpus
echo "$(date -Is) Stage 6" > "${run_dir}/run.status"
bash run_tfgridnet_sms_wsj_1ch_paper.sh --stage 6 --stop_stage 6 >> "${run_dir}/run.log" 2>&1
