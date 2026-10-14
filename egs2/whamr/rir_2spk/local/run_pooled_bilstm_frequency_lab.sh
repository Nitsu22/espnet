#!/usr/bin/env bash
# Fresh BiLSTM frequency ablations; discard the verification model.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script VARIANT GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
variant=$1
gpu_index=$2
result_recipe=$(realpath "$3")
case "$variant" in
    bilstm_post_bimamba|bilstm_post_bilstm|bilstm_both_freq_bilstm) ;;
    *) exit 2 ;;
esac
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
prefix=train_pooled_bimamba_2spk_nf_16k_sweep_v2
config="conf/tuning/${prefix}_${variant}.yaml"
run_dir="$result_recipe/exp/bilstm_frequency_training_20261015/$variant"
stats_dir="$result_recipe/exp/rir_stats_${prefix}_${variant}"
model_dir="$result_recipe/exp/rir_${prefix}_${variant}"
evaluation_dir="$result_recipe/exp/bilstm_frequency_whamr_20261015/$variant"
ls_code_root="$result_recipe/exp/frequency_ls_eval_code_20261014"
[[ -x "$python" && -f "$config" ]] || exit 1
[[ -f "$ls_code_root/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" ]] || exit 1
for marker in "$run_dir" "$model_dir" "$stats_dir" "$evaluation_dir"; do
    [[ ! -e "$marker" ]] || { echo "Existing output: $marker" >&2; exit 1; }
done
mkdir -p "$run_dir/.tmp"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$run_dir/.tmp" NUMBA_CACHE_DIR="$run_dir/.tmp/numba"
export XDG_CACHE_HOME="$run_dir/.tmp/cache" TRITON_CACHE_DIR="$run_dir/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$run_dir/.tmp/torch_extensions"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    printf '%s\n' "$phase" > "$run_dir/final_phase"
    rm -rf -- "$run_dir/.tmp"
    exit "$status"
}
trap finish EXIT
"$python" -B - "$gpu_index" "$run_dir" "$config" "$model_dir" "$variant" <<'PY'
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys

gpu, run_dir, config, model_dir, variant = sys.argv[1:]
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == int(gpu))
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
config = Path(config)
report = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), gpu_uuid=selected[1],
              gpu=selected[2], variant=variant, model_directory=model_dir,
              config=str(config.resolve()),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
(Path(run_dir) / 'started.json').write_text(json.dumps(report, indent=2) + '\n')
PY
phase=gpu_verification
"$python" -B local/check_two_speaker_training.py --config "$config" \
    --data-dir "$result_recipe/dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k" \
    --valid-data-dir "$result_recipe/dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k" \
    --batch-size 4 --warmup-steps 2 --timed-steps 5 \
    --output "$run_dir/gpu_check.json"
phase=input_shapes
"$python" -B local/prepare_input_shapes.py --config "$config" \
    --dump "$result_recipe/dump_nf_2spk_16k_min" --output "$stats_dir"
phase=training
bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
    --stage 6 --stop_stage 6 --resume false --ngpu 1 --python "$python" \
    --dumpdir "$result_recipe/dump_nf_2spk_16k_min" \
    --rir_stats_dir "$stats_dir" --rir_exp "$model_dir"
printf '0\n' > "$run_dir/training_exit_status"
phase=finalization
"$python" -B - "$run_dir" "$model_dir" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

run_dir, model_dir = map(Path, sys.argv[1:])
checkpoint = model_dir / 'valid.loss.best.pth'
config = model_dir / 'config.yaml'
if not checkpoint.is_file() or not config.is_file():
    raise RuntimeError('Best checkpoint or configuration is missing')
report = dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
              checkpoint=str(checkpoint.resolve()),
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
(run_dir / 'training_complete.json').write_text(json.dumps(report, indent=2) + '\n')
PY
phase=whamr_inference_and_physical_scoring
"$python" -B local/evaluate_two_speaker.py --experiment "$model_dir" \
    --data "$result_recipe/dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k" \
    --output "$evaluation_dir/whamr" --device cuda:0
phase=ls_scoring
PYTHONPATH="$ls_code_root" "$python" -B \
    "$ls_code_root/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" \
    --reference_dir "$result_recipe/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_16k" \
    --prediction "$variant" "$evaluation_dir/whamr/inference/rir.scp" \
    --out_dir "$evaluation_dir/ls" --no_comparison_csv
"$python" -B - "$run_dir" "$evaluation_dir" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

run_dir, evaluation_dir = map(Path, sys.argv[1:])
report = json.loads((evaluation_dir / 'ls/comparison.json').read_text())
assert report['count'] == 3000 and report['source_pairs'] == 6000
assert not (evaluation_dir / 'ls/comparison.csv').exists()
(run_dir / 'evaluation_complete.json').write_text(json.dumps(dict(
    finished_at=datetime.datetime.now().astimezone().isoformat(),
    count=report['count'], source_pairs=report['source_pairs'],
    evaluation_directory=str(evaluation_dir),
    reference_preparation_sha256=report['reference_preparation_sha256'],
    comparison_sha256=hashlib.sha256((evaluation_dir / 'ls/comparison.json').read_bytes()).hexdigest(),
), indent=2) + '\n')
PY
phase=complete
