#!/usr/bin/env bash
# Pooled Rec-RIR and BiMamba use matched FLOAT WHAMR inputs and fixed LS references.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
gpu_index=$1
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
result_recipe=$(realpath "$2")
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(git -C "$recipe" rev-parse --show-toplevel)
ls_code="$result_recipe/exp/frequency_ls_eval_code_20261014"
output="$result_recipe/exp/rec_rir_pooled_comparison_20261023"
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
[[ ! -e "$output" && -x "$python" ]] || exit 1
[[ -f "$ls_code/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" ]] || exit 1
mkdir -p "$output/.tmp"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export PYTHONPATH="$code_root" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$output/.tmp" NUMBA_CACHE_DIR="$output/.tmp/numba"
export XDG_CACHE_HOME="$output/.tmp/cache" TRITON_CACHE_DIR="$output/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$output/.tmp/torch_extensions"
phase=validation
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
"$python" -B - "$gpu_index" "$output" "$ls_code" <<'PY'
import csv, datetime, io, json, socket, subprocess, sys
from pathlib import Path

gpu, output, ls_code = sys.argv[1:]
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == int(gpu))
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied')
(Path(output) / 'started.json').write_text(json.dumps(dict(
    started_at=datetime.datetime.now().astimezone().isoformat(), host=socket.gethostname(),
    gpu_index=int(gpu), gpu=selected[2], gpu_uuid=selected[1],
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    ls_code_commit=subprocess.check_output(['git', '-C', ls_code, 'rev-parse', 'HEAD'], text=True).strip()),
    indent=2) + '\n')
PY
phase=evaluation
"$python" -B "$recipe/local/evaluate_rec_rir_pooled_comparison_8k.py" \
    --recipe "$result_recipe" --output "$output" --ls-code "$ls_code"
phase=complete
