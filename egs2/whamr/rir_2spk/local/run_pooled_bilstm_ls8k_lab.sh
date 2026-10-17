#!/usr/bin/env bash
# Ordinary-sweep inference and immutable LS-reference scoring; no retraining.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE' >&2; exit 2; }
gpu_index=$1
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
result_recipe=$(realpath "$2")
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(git -C "$recipe" rev-parse --show-toplevel)
cd "$code_root"
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
ls_code="$result_recipe/exp/frequency_ls_eval_code_20261014"
model="$result_recipe/exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_bilstm_post_bilstm"
data="$result_recipe/dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k"
reference="$result_recipe/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
output="$result_recipe/exp/bilstm_frequency_ls8k_20261017/bilstm_post_bilstm"
[[ ! -e "$output" && -f "$ls_code/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" ]] || exit 1
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
"$python" -B - "$gpu_index" "$output" "$model" "$data" "$reference" <<'PY'
import csv, datetime, hashlib, io, json, shutil, socket, subprocess, sys
from pathlib import Path
import yaml

gpu = int(sys.argv[1])
output, model, data, reference = map(Path, sys.argv[2:])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == gpu)
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied')
config = yaml.safe_load((model / 'config.yaml').read_text())
assert config['sample_rate'] == config['model_conf']['sr'] == 8000
assert config['model_conf']['predictor_conf']['time_module'] == 'bilstm'
assert config['model_conf']['predictor_conf']['post_frequency_module'] == 'bilstm'
preparation = json.loads((reference / 'preparation.json').read_text())
assert preparation['full_dataset'] and preparation['count'] == 3000
assert preparation['config']['sample_rate'] == 8000
assert preparation['config']['rir_length'] == 16000
assert digest(data / 'wav.scp') == preparation['source_index_sha256']['wav.scp']
training = model.parent / 'bilstm_frequency_training_20261016_8k/bilstm_post_bilstm'
assert (training / 'exit_status').read_text().strip() == '0'
assert digest(model / 'valid.loss.best.pth') == json.loads(
    (training / 'training_complete.json').read_text())['checkpoint_sha256']
prediction = output / 'prediction'
prediction.mkdir()
for filename in ('config.yaml', 'valid.loss.best.pth'):
    shutil.copy2(model / filename, prediction / filename)
shutil.copy2(data / 'wav.scp', prediction / 'wav.scp')
metadata = dict(inference='ordinary sweep PIM', sample_rate=8000,
    rir_length=16000, count=3000, input='full utterance, channel 0, canonical FLOAT NF mixture',
    experiment=str(model), checkpoint='valid.loss.best.pth',
    checkpoint_sha256=digest(prediction / 'valid.loss.best.pth'),
    config_sha256=digest(prediction / 'config.yaml'),
    input_index_sha256=digest(prediction / 'wav.scp'))
(prediction / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
(output / 'started.json').write_text(json.dumps(dict(
    started_at=datetime.datetime.now().astimezone().isoformat(),
    host=socket.gethostname(), gpu_index=gpu, gpu=selected[2], gpu_uuid=selected[1],
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    reference_preparation_sha256=digest(reference / 'preparation.json'),
    model_directory=str(model)), indent=2) + '\n')
PY
phase=inference
"$python" -B -m espnet2.bin.rec_rir_pit_inference \
    --train_config "$output/prediction/config.yaml" \
    --model_file "$output/prediction/valid.loss.best.pth" \
    --wav_scp "$output/prediction/wav.scp" --output_dir "$output/prediction/inference" \
    --sample_rate 8000 --rir_length 16000 --output_subtype FLOAT --device cuda:0
phase=ls_scoring
PYTHONPATH="$ls_code" "$python" -B \
    "$ls_code/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" \
    --reference_dir "$reference" \
    --prediction BiLSTM_time_light_post_BiLSTM_8k "$output/prediction/inference/rir.scp" \
    --out_dir "$output/ls" --no_comparison_csv
"$python" -B - "$output" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
output = Path(sys.argv[1])
report = json.loads((output / 'ls/comparison.json').read_text())
assert report['count'] == 3000 and report['source_pairs'] == 6000
assert not (output / 'ls/comparison.csv').exists()
(output / 'evaluation_complete.json').write_text(json.dumps(dict(
    finished_at=datetime.datetime.now().astimezone().isoformat(), count=3000,
    source_pairs=6000, reference_preparation_sha256=report['reference_preparation_sha256'],
    comparison_sha256=hashlib.sha256((output / 'ls/comparison.json').read_bytes()).hexdigest()),
    indent=2) + '\n')
PY
phase=complete
