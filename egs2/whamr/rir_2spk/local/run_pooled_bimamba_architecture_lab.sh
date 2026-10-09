#!/usr/bin/env bash
# One isolated architecture experiment on one explicitly selected lab GPU.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script VARIANT GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
variant=$1
gpu_index=$2
result_recipe=$(realpath "$3")
case "$variant" in time2_freq4|time4_freq2|scalar_pool|local_freq_only) ;; *) exit 2 ;; esac
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
prefix=train_pooled_bimamba_2spk_nf_16k_sweep_v2
config="conf/tuning/${prefix}_${variant}.yaml"
run_dir="$result_recipe/exp/architecture_ablation_training_20261009/$variant"
stats_dir="$result_recipe/exp/rir_stats_${prefix}_${variant}"
model_dir="$result_recipe/exp/rir_${prefix}_${variant}"
[[ -d "$run_dir" ]] || mkdir -p "$run_dir"
[[ ! -e "$run_dir/started.json" && ! -e "$run_dir/exit_status" && ! -e "$model_dir/checkpoint.pth" ]] || {
    echo 'Existing run: select a fresh result directory or resume explicitly' >&2; exit 1;
}
export CUDA_VISIBLE_DEVICES="$gpu_index"
export NUMBA_CACHE_DIR="/tmp/nitsu-bimamba-architecture-${variant}-numba"
export OMP_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
trap 'echo $? > "$run_dir/exit_status"' EXIT
"$python" - "$run_dir" "$config" "$gpu_index" "$model_dir" <<'PY'
import datetime, hashlib, json, pathlib, socket, subprocess, sys
run_dir, config, gpu, model_dir = sys.argv[1:]
p = pathlib.Path(config)
report = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), config=str(p.resolve()),
              config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              model_directory=model_dir)
(pathlib.Path(run_dir) / 'started.json').write_text(json.dumps(report, indent=2) + '\n')
PY
"$python" local/prepare_input_shapes.py --config "$config" \
    --dump dump_nf_2spk_16k_min --output "$stats_dir"
"$python" local/check_two_speaker_training.py --config "$config" \
    --data-dir dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k \
    --valid-data-dir dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k \
    --batch-size 4 --warmup-steps 1 --timed-steps 2 --output "$run_dir/gpu_check.json"
bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
    --stage 6 --stop_stage 6 --ngpu 1 --python "$python" \
    --rir_stats_dir "$stats_dir" --rir_exp "$model_dir"
