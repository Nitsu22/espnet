#!/usr/bin/env bash
# CPU-only recovery of aggregation after successful inference and scoring.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: script RESULT_RECIPE' >&2; exit 2; }
result_recipe=$(realpath "$1")
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(git -C "$recipe" rev-parse --show-toplevel)
output="$result_recipe/exp/rec_rir_pooled_comparison_20261023"
ls_code="$result_recipe/exp/frequency_ls_eval_code_20261014"
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
[[ -f "$output/ls/comparison.json" && ! -e "$output/evaluation_complete.json" ]] || exit 1
[[ $(cat "$output/exit_status") == 1 && ! -e "$output/first_attempt_exit_status" ]] || exit 1
cp "$output/exit_status" "$output/first_attempt_exit_status"
cp "$output/final_phase" "$output/first_attempt_final_phase"
mkdir -p "$output/.tmp"
export CUDA_VISIBLE_DEVICES='' TZ=Asia/Tokyo
export PYTHONPATH="$code_root" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$output/.tmp" NUMBA_CACHE_DIR="$output/.tmp/numba"
export XDG_CACHE_HOME="$output/.tmp/cache" TORCH_EXTENSIONS_DIR="$output/.tmp/torch"
phase=finalization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$output/exit_status"
    printf '%s\n' "$phase" > "$output/final_phase"
    rm -rf -- "$output/.tmp"
    exit "$status"
}
trap finish EXIT
cd "$code_root"
"$python" -B - "$output" <<'PY'
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

output = Path(sys.argv[1])
(output / 'finalization_started.json').write_text(json.dumps(dict(
    started_at=datetime.datetime.now().astimezone().isoformat(),
    host=socket.gethostname(), device='CPU; reuse completed inference and scoring',
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()), indent=2) + '\n')
PY
"$python" -B "$recipe/local/evaluate_rec_rir_pooled_comparison_8k.py" \
    --recipe "$result_recipe" --output "$output" --ls-code "$ls_code" --finalize-only
phase=complete
