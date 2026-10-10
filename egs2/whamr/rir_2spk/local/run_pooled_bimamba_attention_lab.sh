#!/usr/bin/env bash
# Scratch training followed by WHAMR-only evaluation on one selected lab GPU.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script VARIANT GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
variant=$1
gpu_index=$2
result_recipe=$(realpath "$3")
case "$variant" in post_speaker|post_joint|post_frequency|pre_speaker) ;; *) exit 2 ;; esac
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
recipe=$(cd "$(dirname "$0")/.." && pwd)
cd "$recipe"
code_root=$(git rev-parse --show-toplevel)
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
prefix=train_pooled_bimamba_2spk_nf_16k_sweep_v2
config="conf/tuning/${prefix}_${variant}.yaml"
run_dir="$result_recipe/exp/attention_ablation_training_20261011/$variant"
stats_dir="$result_recipe/exp/rir_stats_${prefix}_${variant}"
model_dir="$result_recipe/exp/rir_${prefix}_${variant}"
eval_dir="$result_recipe/exp/attention_ablation_whamr_20261011/$variant"
test_data=$(realpath dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k)
[[ -x "$python" && -f "$config" ]] || exit 1
# The caller may create directories for launch.log; all state and model outputs
# must still be fresh. Never silently overwrite a run or resume a checkpoint.
for marker in "$run_dir/started.json" "$run_dir/exit_status" \
    "$model_dir" "$stats_dir" "$eval_dir/started.json" \
    "$eval_dir/exit_status" "$eval_dir/smoke_whamr" "$eval_dir/whamr"; do
    [[ ! -e "$marker" ]] || { echo "Existing experiment output: $marker" >&2; exit 1; }
done
mkdir -p "$run_dir" "$eval_dir"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export TZ=Asia/Tokyo
export NUMBA_CACHE_DIR="/tmp/nitsu-bimamba-attention-${variant}-numba"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    case "$phase" in
        training) printf '%s\n' "$status" > "$run_dir/training_exit_status" ;;
        evaluation_smoke|inference|scoring|evaluation_finalization|complete)
            printf '%s\n' "$status" > "$eval_dir/exit_status" ;;
    esac
    case "$phase" in
        inference) printf '%s\n' "$status" > "$eval_dir/inference_exit_status" ;;
        scoring) printf '%s\n' "$status" > "$eval_dir/scoring_exit_status" ;;
    esac
    "$python" - "$run_dir" "$phase" "$status" <<'PY' || true
import datetime, json, pathlib, sys
root, phase, status = sys.argv[1:]
root = pathlib.Path(root)
now = datetime.datetime.now().astimezone()
start_path = root / 'started.json'
start = json.loads(start_path.read_text()) if start_path.exists() else {}
start_time = datetime.datetime.fromisoformat(start.get('started_at', now.isoformat()))
report = dict(finished_at=now.isoformat(), phase=phase,
              exit_status=int(status), elapsed_seconds=(now-start_time).total_seconds())
if int(status):
    report['failed_phase'] = phase
(root / 'pipeline_status.json').write_text(json.dumps(report, indent=2) + '\n')
PY
    exit "$status"
}
trap finish EXIT
set_phase() {
    phase=$1
    "$python" - "$run_dir" "$phase" <<'PY'
import datetime, json, pathlib, sys
root, phase = sys.argv[1:]
report = dict(phase=phase, phase_started_at=datetime.datetime.now().astimezone().isoformat())
(pathlib.Path(root) / 'phase.json').write_text(json.dumps(report, indent=2) + '\n')
with (pathlib.Path(root) / 'phase_history.jsonl').open('a') as stream:
    stream.write(json.dumps(report) + '\n')
PY
}
"$python" - "$run_dir" "$eval_dir" "$config" "$gpu_index" "$model_dir" "$test_data" "$variant" <<'PY'
import datetime, hashlib, json, pathlib, socket, subprocess, sys
run_dir, eval_dir, config, gpu, model_dir, data, variant = sys.argv[1:]
data = pathlib.Path(data)
maps = []
for name in ('wav.scp', 'rir_ref1.scp', 'rir_ref2.scp'):
    rows = [line.split(maxsplit=1) for line in (data/name).read_text().splitlines() if line.strip()]
    if len(rows) != 3000 or len({row[0] for row in rows}) != 3000:
        raise ValueError(f'Expected 3000 unique WHAMR test examples: {name}')
    maps.append({row[0] for row in rows})
if any(ids != maps[0] for ids in maps[1:]):
    raise ValueError('WHAMR mixture and RIR reference IDs differ')
p = pathlib.Path(config)
report = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
              host=socket.gethostname(), gpu_index=int(gpu), variant=variant,
              config=str(p.resolve()), config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
              commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              model_directory=model_dir, test_data=str(data), expected_utterances=3000,
              evaluation_directory=eval_dir, evaluation_scope='WHAMR only')
(pathlib.Path(run_dir) / 'started.json').write_text(json.dumps(report, indent=2) + '\n')
PY
set_phase input_shapes
"$python" local/prepare_input_shapes.py --config "$config" \
    --dump dump_nf_2spk_16k_min --output "$stats_dir"
set_phase gpu_check
"$python" local/check_two_speaker_training.py --config "$config" \
    --data-dir dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k \
    --valid-data-dir dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k \
    --batch-size 4 --warmup-steps 1 --timed-steps 2 --output "$run_dir/gpu_check.json"
"$python" - "$run_dir/gpu_check.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
if report['parameters'] != 489964:
    raise ValueError(f"Unexpected attention-ablation parameter count: {report['parameters']}")
for flag in ('teacher_permutation_invariant', 'noise_free_mixture_verified',
             'active_parameters_have_finite_gradients'):
    if report[flag] is not True:
        raise ValueError(f'GPU check failed: {flag}')
if report['rir_shape'] != [2, 32000]:
    raise ValueError('GPU inference did not return two two-second RIRs')
PY
set_phase training
bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
    --stage 6 --stop_stage 6 --resume false --ngpu 1 --python "$python" \
    --rir_stats_dir "$stats_dir" --rir_exp "$model_dir"
printf '0\n' > "$run_dir/training_exit_status"
set_phase training_finalization
"$python" - "$run_dir" "$eval_dir" "$model_dir" <<'PY'
import datetime, hashlib, json, pathlib, sys
root, eval_dir, model_dir = map(pathlib.Path, sys.argv[1:])
checkpoint = model_dir/'valid.loss.best.pth'
if not checkpoint.is_file() or not (model_dir/'config.yaml').is_file():
    raise ValueError('Successful training must save its selected best checkpoint and config')
started = json.loads((root/'started.json').read_text())
now = datetime.datetime.now().astimezone()
phases = [json.loads(line) for line in (root/'phase_history.jsonl').read_text().splitlines()]
training_start = next(p['phase_started_at'] for p in phases if p['phase'] == 'training')
report = dict(finished_at=now.isoformat(), commit=started['commit'],
              parameters=json.loads((root/'gpu_check.json').read_text())['parameters'],
              training_seconds=(now-datetime.datetime.fromisoformat(training_start)).total_seconds(),
              model_directory=str(model_dir), checkpoint=str(checkpoint),
              selected_checkpoint=str(checkpoint.resolve()),
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              trained_config_sha256=hashlib.sha256((model_dir/'config.yaml').read_bytes()).hexdigest(),
              source_config_sha256=started['config_sha256'])
(root/'training_complete.json').write_text(json.dumps(report, indent=2)+'\n')
evaluation = dict(started)
evaluation['started_at'] = now.isoformat()
(eval_dir/'started.json').write_text(json.dumps(evaluation, indent=2)+'\n')
PY
set_phase evaluation_smoke
"$python" local/evaluate_two_speaker.py --stage all --limit 2 \
    --experiment "$model_dir" --data "$test_data" --output "$eval_dir/smoke_whamr"
printf '0\n' > "$eval_dir/smoke_exit_status"
set_phase inference
"$python" local/evaluate_two_speaker.py --stage inference \
    --experiment "$model_dir" --data "$test_data" --output "$eval_dir/whamr"
printf '0\n' > "$eval_dir/inference_exit_status"
# Inference subprocesses have exited. Scoring uses CPU only and bounded threads.
export CUDA_VISIBLE_DEVICES=
set_phase scoring
"$python" local/evaluate_two_speaker.py --stage score \
    --experiment "$model_dir" --data "$test_data" --output "$eval_dir/whamr"
printf '0\n' > "$eval_dir/scoring_exit_status"
set_phase evaluation_finalization
"$python" local/finalize_attention_whamr.py \
    --training "$run_dir" --output "$eval_dir" --expected-utts 3000
set_phase complete
