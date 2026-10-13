#!/usr/bin/env bash
# Exact +/-62-frame local BiMamba, then unchanged global BiMamba and sweep PIT.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
gpu_index=$1
result_recipe=$(realpath "$2")
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
variant=time_local_bimamba125
prefix=train_pooled_bimamba_2spk_nf_16k_sweep_v2
config="conf/tuning/${prefix}_${variant}.yaml"
run_dir="$result_recipe/exp/time_local_bimamba_training_20261014/$variant"
stats_dir="$result_recipe/exp/rir_stats_${prefix}_${variant}"
model_dir="$result_recipe/exp/rir_${prefix}_${variant}"
[[ -x "$python" && -f "$config" ]] || exit 1
case "$run_dir" in /net/midgar/work2/nitsu/*) ;; *) exit 2 ;; esac
for marker in "$run_dir" "$model_dir" "$stats_dir"; do
    [[ ! -e "$marker" ]] || { echo "Existing output: $marker" >&2; exit 1; }
done
mkdir -p "$run_dir/tmp"
export TMPDIR="$run_dir/tmp" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$TMPDIR/cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export NUMBA_CACHE_DIR="$TMPDIR/numba" TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions" CUDA_CACHE_PATH="$TMPDIR/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    printf '%s\n' "$phase" > "$run_dir/final_phase"
    # This fresh private temporary directory belongs only to this invocation.
    rm -rf -- "$TMPDIR"
    exit "$status"
}
trap finish EXIT
"$python" - "$gpu_index" "$run_dir" "$config" "$model_dir" <<'PY'
import csv,datetime,hashlib,io,json,pathlib,socket,subprocess,sys
gpu, run_dir, config, model_dir = sys.argv[1:]
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == int(gpu))
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True))))
if int(selected[2]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied; refusing to start')
config = pathlib.Path(config)
report = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), gpu_uuid=selected[1],
              model_directory=model_dir, config=str(config.resolve()),
              config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
(pathlib.Path(run_dir)/'started.json').write_text(json.dumps(report, indent=2)+'\n')
PY
phase=cuda_window_reference
"$python" local/check_time_local_bimamba_cuda.py --output "$run_dir/cuda_window_reference.json"
phase=input_shapes
"$python" local/prepare_input_shapes.py --config "$config" \
    --dump dump_nf_2spk_16k_min --output "$stats_dir"
phase=gpu_check
"$python" local/check_two_speaker_training.py --config "$config" \
    --data-dir dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k \
    --valid-data-dir dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k \
    --batch-size 4 --warmup-steps 1 --timed-steps 2 --output "$run_dir/gpu_check.json"
"$python" - "$run_dir/gpu_check.json" <<'PY'
import json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
if report['rir_shape'] != [2,32000] or report['parameters'] != 751084:
    raise RuntimeError('Unexpected RIR output or model size')
for flag in ('teacher_permutation_invariant', 'noise_free_mixture_verified',
             'active_parameters_have_finite_gradients'):
    if report[flag] is not True:
        raise RuntimeError('CUDA validation failed: '+flag)
PY
phase=training
bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
    --stage 6 --stop_stage 6 --resume false --ngpu 1 --python "$python" \
    --rir_stats_dir "$stats_dir" --rir_exp "$model_dir" --rir_args '--valid_batch_size 1 --log_interval 50'
printf '0\n' > "$run_dir/training_exit_status"
phase=finalization
"$python" - "$run_dir" "$model_dir" <<'PY'
import datetime,hashlib,json,pathlib,sys
run_dir, model_dir = map(pathlib.Path, sys.argv[1:])
checkpoint, config = model_dir/'valid.loss.best.pth', model_dir/'config.yaml'
if not checkpoint.is_file() or not config.is_file():
    raise RuntimeError('Training did not save the selected checkpoint/config')
report=dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
(run_dir/'training_complete.json').write_text(json.dumps(report, indent=2)+'\n')
PY
phase=complete
