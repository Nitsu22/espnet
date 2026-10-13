#!/usr/bin/env bash
# Scratch baseline training using the existing two-speaker noisy mixtures.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
variant=noisy_baseline
gpu_index=$1
result_recipe=$(realpath "$2")
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
prefix=train_pooled_bimamba_2spk_noisy_16k_sweep_v2
config="conf/tuning/${prefix}.yaml"
run_dir="$result_recipe/exp/noisy_baseline_training_20261013"
stats_dir="$result_recipe/exp/rir_stats_${prefix}"
model_dir="$result_recipe/exp/rir_${prefix}"
[[ -x "$python" && -f "$config" ]] || exit 1
for marker in "$run_dir/started.json" "$run_dir/exit_status" "$model_dir" "$stats_dir"; do
    [[ ! -e "$marker" ]] || { echo "Existing experiment output: $marker" >&2; exit 1; }
done
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="/tmp/nitsu-frequency-${variant}-numba"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    printf '%s\n' "$phase" > "$run_dir/final_phase"
    exit "$status"
}
trap finish EXIT
"$python" - "$gpu_index" "$run_dir" "$config" "$model_dir" "$variant" <<'PY'
import csv
import datetime
import hashlib
import io
import json
import pathlib
import socket
import subprocess
import sys

gpu, run_dir, config, model_dir, variant = sys.argv[1:]
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == int(gpu))
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[2]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
config = pathlib.Path(config)
report = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), gpu_uuid=selected[1],
              variant=variant, model_directory=model_dir,
              config=str(config.resolve()),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
(pathlib.Path(run_dir) / 'started.json').write_text(json.dumps(report, indent=2) + '\n')
PY
phase=gpu_check
"$python" local/check_two_speaker_training.py --config "$config" --noise \
    --data-dir dump_noisy_2spk_16k_min/raw/tr_rir_2spk_noisy_min_16k \
    --valid-data-dir dump_noisy_2spk_16k_min/raw/cv_rir_2spk_noisy_min_16k \
    --batch-size 4 --warmup-steps 1 --timed-steps 2 --output "$run_dir/gpu_check.json"
"$python" - "$run_dir/gpu_check.json" <<'PY_CHECK'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['parameters'] == 456428
assert report['noisy_mixture_verified'] is True
assert report['noise_free_mixture_verified'] is False
assert report['active_parameters_have_finite_gradients'] is True
assert report['teacher_permutation_invariant'] is True
assert report['rir_shape'] == [2, 32000]
PY_CHECK
phase=training
bash run_pooled_bimamba_2spk_noisy_16k_sweep_v2.sh \
    --stage 5 --stop_stage 6 --resume false --ngpu 1 --python "$python" \
    --rir_stats_dir "$stats_dir" --rir_exp "$model_dir"
printf '0\n' > "$run_dir/training_exit_status"
phase=finalization
"$python" - "$run_dir" "$model_dir" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

run_dir, model_dir = map(pathlib.Path, sys.argv[1:])
checkpoint = model_dir / 'valid.loss.best.pth'
config = model_dir / 'config.yaml'
if not checkpoint.is_file() or not config.is_file():
    raise RuntimeError('Training did not save the selected best checkpoint and config')
report = dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
              checkpoint=str(checkpoint.resolve()),
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
(run_dir / 'training_complete.json').write_text(json.dumps(report, indent=2) + '\n')
PY
phase=complete
