#!/usr/bin/env bash
# Scratch early-split BiLSTM CTF training, with GPU checks and completion records.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
gpu_index=$1
variant=split_bilstm_enc4_dec4_post4_mse
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX must be an integer' >&2; exit 2; }
result_recipe=$(realpath "$2")
output_root=$(realpath -m "$result_recipe/exp")
case "$output_root" in
    /net/midgar/work2/nitsu/*|/gs/bs/tga-shinoda/nitsu/*) ;;
    *) echo 'Results must resolve inside an authorized nitsu directory' >&2; exit 1 ;;
esac
[[ -d "$result_recipe" && -O "$result_recipe" ]] || {
    echo 'RESULT_RECIPE must be an existing directory owned by the current user' >&2; exit 1;
}
[[ ! -e "$output_root" || ( -d "$output_root" && -O "$output_root" ) ]] || {
    echo 'The results exp directory must be owned by the current user' >&2; exit 1;
}
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
prefix=train_split_bilstm_2spk_nf_8k_sweep_v2_enc4_dec4_post4_rir_mse
run_tag="${variant}_$(hostname -s)_20261027"
config="conf/tuning/${prefix}.yaml"
run_dir="$output_root/${run_tag}_job"
stats_dir="$output_root/rir_stats_${run_tag}"
model_dir="$output_root/rir_${run_tag}"
dump_dir="$result_recipe/dump_nf_2spk_8k_min"
[[ -x "$python" && -f "$config" && -f "$dump_dir/preparation.json" ]] || {
    echo 'Python, configuration, or prepared 8 kHz dump is missing' >&2; exit 1;
}
for marker in "$run_dir" "$model_dir" "$stats_dir"; do
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        echo "Existing output: $marker" >&2; exit 1;
    }
done
mkdir -p "$output_root"
mkdir "$run_dir"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    printf '%s\n' "$phase" > "$run_dir/final_phase"
    if [[ "$phase" == training ]]; then
        printf '%s\n' "$status" > "$run_dir/training_exit_status"
    fi
    rm -rf -- "$run_dir/.tmp"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$run_dir/.tmp"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$run_dir/.tmp" NUMBA_CACHE_DIR="$run_dir/.tmp/numba"
export XDG_CACHE_HOME="$run_dir/.tmp/cache" TRITON_CACHE_DIR="$run_dir/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$run_dir/.tmp/torch_extensions"
export CUDA_CACHE_PATH="$run_dir/.tmp/cuda" TORCH_HOME="$run_dir/.tmp/torch"
export MPLCONFIGDIR="$run_dir/.tmp/matplotlib"

assert_free_gpu() {
    "$python" -B - "$gpu_index" "$1" "$config" "$model_dir" "$phase" "$variant" <<'PY'
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys

gpu, report_path, config, model_dir, phase, variant = sys.argv[1:]
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next(([v.strip() for v in row] for row in rows
                 if row and int(row[0]) == int(gpu)), None)
if selected is None:
    raise RuntimeError(f'GPU {gpu} does not exist on this host')
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
config = Path(config)
report = dict(checked_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), gpu_uuid=selected[1],
              gpu=selected[2], memory_used_mib=int(selected[3]), phase=phase,
              variant=variant, model_directory=model_dir,
              config=str(config.resolve()),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
PY
}

phase=gpu_verification
assert_free_gpu "$run_dir/started.json"
"$python" -B local/check_two_speaker_training.py --config "$config" \
    --data-dir "$dump_dir/raw/tr_rir_2spk_nf_min_8k" \
    --valid-data-dir "$dump_dir/raw/cv_rir_2spk_nf_min_8k" \
    --batch-size 4 --warmup-steps 2 --timed-steps 5 \
    --output "$run_dir/gpu_check.json"
phase=input_shapes
# Stage 5 input-only batching statistics, with prepared-manifest hash checks.
"$python" -B local/prepare_input_shapes.py --config "$config" \
    --dump "$dump_dir" --output "$stats_dir"
phase=training_gpu_verification
assert_free_gpu "$run_dir/training_gpu.json"
phase=training
bash run_split_bilstm_2spk_nf_8k_sweep_v2.sh \
    --stage 6 --stop_stage 6 --resume false --ngpu 1 --python "$python" \
    --dumpdir "$dump_dir" --rir_stats_dir "$stats_dir" --rir_exp "$model_dir"
printf '0\n' > "$run_dir/training_exit_status"
phase=finalization
"$python" -B - "$run_dir" "$model_dir" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

import yaml

run_dir, model_dir = map(Path, sys.argv[1:])
config = model_dir / 'config.yaml'
if not config.is_file():
    raise RuntimeError('Saved training configuration is missing')
settings = yaml.safe_load(config.read_text())
split, metric, _ = settings['best_model_criterion'][0]
checkpoint = model_dir / f'{split}.{metric}.best.pth'
if not checkpoint.is_file():
    raise RuntimeError(f'Best checkpoint is missing: {checkpoint}')
report = dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
              checkpoint=str(checkpoint.resolve()),
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              selection_criterion=settings['best_model_criterion'][0],
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
(run_dir / 'training_complete.json').write_text(json.dumps(report, indent=2) + '\n')
PY
phase=complete
