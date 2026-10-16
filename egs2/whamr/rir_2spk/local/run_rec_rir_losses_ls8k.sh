#!/usr/bin/env bash
set -euo pipefail
workspace=${1:?workspace root required}
output=${2:?fresh output directory required}
gpu=${3:?host GPU index required}
task_tmp=${4:?dedicated owned /tmp/nitsu directory required}
[[ "$gpu" =~ ^[0-9]+$ && "$task_tmp" == /tmp/nitsu/rec-rir-loss-ls8k.* && -d "$task_tmp" && -O "$task_tmp" ]] || exit 2
[[ ! -e "$output" && ! -e "$output.exit_status" ]] || exit 2
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
python=${PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
export TMPDIR="$task_tmp" TZ=Asia/Tokyo PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$TMPDIR/cache" NUMBA_CACHE_DIR="$TMPDIR/numba"
export XDG_CONFIG_HOME="$TMPDIR/config" MPLCONFIGDIR="$TMPDIR/matplotlib"
export TORCH_HOME="$TMPDIR/torch" CUDA_CACHE_PATH="$TMPDIR/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton" TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$output.exit_status"
    date '+%Y-%m-%d %H:%M:%S' > "$output.finished_at"
    rm -rf -- "$task_tmp"
    exit "$status"
}
trap finish EXIT
cd "$repo"
date '+%Y-%m-%d %H:%M:%S' > "$output.started_at"
hostname > "$output.host"
git rev-parse HEAD > "$output.git_commit"
# Check again immediately before loading either model.
"$python" -B - "$gpu" <<'PY'
import csv
import io
import subprocess
import sys

rows = csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)))
chosen = next([v.strip() for v in row] for row in rows if int(row[0]) == int(sys.argv[1]))
apps = csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)))
if int(chosen[2]) >= 200 or any(row and row[0].strip() == chosen[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
PY
nvidia-smi --id="$gpu" --query-gpu=index,uuid,name,memory.total --format=csv > "$output.gpu.csv"
"$python" -B egs2/whamr/rir_2spk/local/evaluate_rec_rir_losses_ls8k.py \
    --workspace "$workspace" --output "$output" --device cuda:0
