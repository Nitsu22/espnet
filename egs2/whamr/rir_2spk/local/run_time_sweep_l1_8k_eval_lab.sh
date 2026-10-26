#!/usr/bin/env bash
# One ordinary-PIM inference pass, scored against physical WHAMR and LS RIRs.
set -euo pipefail
[[ $# == 2 || $# == 3 ]] || { echo 'Usage: script GPU_INDEX RESULT_RECIPE [time_l1|baseline|rec_rir|baseline_ft45_plateau|baseline_scratch_plateau|bilstm_pre8_post4_time_l1_plateau]' >&2; exit 2; }
gpu_index=$1
[[ "$gpu_index" =~ ^[0-9]+$ ]] || exit 2
variant=${3:-time_l1}
case "$variant" in time_l1|baseline|rec_rir|baseline_ft45_plateau|baseline_scratch_plateau|bilstm_pre8_post4_time_l1_plateau) ;; *) echo 'Unsupported variant' >&2; exit 2 ;; esac
prediction_label="BiMamba_8k_$variant"
[[ "$variant" != rec_rir ]] || prediction_label=RecRIR_8k_3loss
result_recipe=$(realpath "$2")
case "$result_recipe" in
    /net/midgar/work2/nitsu/*|/gs/bs/tga-shinoda/nitsu/*) ;;
    *) echo 'Use an authorized nitsu results directory' >&2; exit 1 ;;
esac
[[ -d "$result_recipe" && -O "$result_recipe" && -O "$result_recipe/exp" ]] || exit 1
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(git -C "$recipe" rev-parse --show-toplevel)
cd "$code_root"
python=${RIR_LAB_PYTHON:-/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python}
scoring_root="$result_recipe/exp/frequency_ls_eval_code_20261014"
expected_scorer=4e8a0f5b9b0bc5d73e3e5ace5b42d4b29b300f72
[[ "$(git -C "$scoring_root" rev-parse HEAD)" == "$expected_scorer" ]] || exit 1
[[ -z "$(git -C "$scoring_root" status --porcelain)" ]] || exit 1
if [[ "$variant" == rec_rir ]]; then
    model=$(realpath "$result_recipe/../rir/exp/rir_train_rec_rir_2spk_pit_clean_8192")
    training="$result_recipe/exp/rec_rir_ls8k_20261015"
    output="$result_recipe/exp/time_sweep_l1_recrir_eval8k_20261019"
elif [[ "$variant" == baseline ]]; then
    model="$result_recipe/exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2"
    training="$result_recipe/exp/pooled_bimamba_8k_launch_20261009"
    output="$result_recipe/exp/time_sweep_l1_baseline_eval8k_20261019"
elif [[ "$variant" == baseline_ft45_plateau ]]; then
    model="$result_recipe/exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_ft45_plateau_lr1em3"
    training="$result_recipe/exp/baseline_plateau_ft45_lr1em3_training_20261021"
    output="$result_recipe/exp/baseline_plateau_ft45_lr1em3_eval8k_20261023"
elif [[ "$variant" == baseline_scratch_plateau ]]; then
    model="$result_recipe/exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_scratch_plateau_200ep"
    training="$result_recipe/exp/baseline_scratch_plateau_200ep_training_20261022"
    output="$result_recipe/exp/baseline_scratch_plateau_eval8k_20261026"
elif [[ "$variant" == bilstm_pre8_post4_time_l1_plateau ]]; then
    model="$result_recipe/exp/rir_train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4_time_l1_plateau_200ep"
    training="$result_recipe/exp/bilstm_pre8_post4_time_l1_training_20261023"
    output="$result_recipe/exp/bilstm_pre8_post4_time_l1_plateau_eval8k_20261026"
    prediction_label=BiLSTM_pre8_post4_time_L1_8k
else
    model="$result_recipe/exp/rir_train_pooled_bimamba_2spk_nf_8k_sweep_v2_time_l1"
    training="$result_recipe/exp/time_sweep_l1_training_20261019"
    output="$result_recipe/exp/time_sweep_l1_eval8k_20261019"
fi
data="$result_recipe/dump_nf_2spk_8k_min/raw/tt_rir_2spk_nf_min_8k"
reference="$result_recipe/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k"
[[ ! -e "$output" && ! -L "$output" ]] || { echo "Existing output: $output" >&2; exit 1; }
mkdir "$output"
phase=initialization
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$output/exit_status"
    printf '%s\n' "$phase" > "$output/final_phase"
    rm -rf -- "$output/.tmp"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$output/.tmp"
export CUDA_VISIBLE_DEVICES="$gpu_index" TZ=Asia/Tokyo
export PYTHONPATH="$code_root" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$output/.tmp" NUMBA_CACHE_DIR="$output/.tmp/numba"
export XDG_CACHE_HOME="$output/.tmp/cache" TRITON_CACHE_DIR="$output/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$output/.tmp/torch_extensions"
export CUDA_CACHE_PATH="$output/.tmp/cuda" TORCH_HOME="$output/.tmp/torch"
export MPLCONFIGDIR="$output/.tmp/matplotlib"
phase=validation
"$python" -B - "$gpu_index" "$output" "$model" "$training" "$data" "$reference" "$scoring_root" "$variant" <<'PY'
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import numpy as np
import yaml

from espnet2.rir.rec_rir.pim import RecRIRPIM

gpu = int(sys.argv[1])
output, model, training, data, reference, scoring_root = map(Path, sys.argv[2:8])
variant = sys.argv[8]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True))))
selected = next([v.strip() for v in row] for row in rows if int(row[0]) == gpu)
apps = list(csv.reader(io.StringIO(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'], text=True))))
if int(selected[3]) >= 200 or any(row and row[0].strip() == selected[1] for row in apps):
    raise RuntimeError('Selected GPU is occupied')
if variant == 'rec_rir':
    prior = json.loads((training / 'complete.json').read_text())
    assert prior['count'] == 3000 and prior['source_pairs'] == 6000
    completed = json.loads((training / 'metadata.json').read_text())
    assert Path(completed['experiment']).resolve() == model.resolve()
    checkpoint = model / '97epoch.pth'
elif variant == 'baseline':
    assert (training / 'exit_status').read_text().strip() == '0'
    completed = json.loads((model.parent / 'damsep_bimamba_ls8k_20261015/bimamba/metadata.json').read_text())
    assert Path(completed['experiment']).resolve() == model.resolve()
    checkpoint = model / 'valid.loss.best.pth'
else:
    assert (training / 'exit_status').read_text().strip() == '0'
    assert (training / 'final_phase').read_text().strip() == 'complete'
    completed = json.loads((training / 'training_complete.json').read_text())
    checkpoint = model / ('valid.loss.best.pth' if variant in ('baseline_ft45_plateau', 'baseline_scratch_plateau')
                          else 'valid.loss_sweep_min.best.pth')
    assert checkpoint.resolve() == Path(completed['checkpoint']).resolve()
assert digest(checkpoint) == completed['checkpoint_sha256']
assert digest(model / 'config.yaml') == completed['config_sha256']
config = yaml.safe_load((model / 'config.yaml').read_text())
assert config['sample_rate'] == config['model_conf']['sr'] == 8000
assert config['model_conf'].get('time_sweep_loss_weight', 0.0) == (1.0 if variant in ('time_l1', 'bilstm_pre8_post4_time_l1_plateau') else 0.0)
if variant == 'rec_rir':
    assert config['rir_model_type'] == 'rec_rir_pit'
    assert all(config['model_conf'][key] == 1.0 for key in ('loss_w_cln', 'loss_w_rvb', 'loss_w_rec'))
    assert config['model_conf']['dim_output_CTF'] // (2 * config['model_conf']['num_spk']) == 60
else:
    assert config['model_conf']['ctf_taps'] == 60
assert config['model_conf']['pim_sweep_duration'] == 8.192
preparation = json.loads((reference / 'preparation.json').read_text())
assert preparation['full_dataset'] and preparation['count'] == 3000
assert preparation['config']['sample_rate'] == 8000
assert preparation['config']['rir_length'] == 16000
pim = RecRIRPIM(8000, config['model_conf']['pim_sweep_duration'])
sweep_hash = hashlib.sha256(pim.sinesweep.numpy().astype(np.float64).tobytes()).hexdigest()
inverse_hash = hashlib.sha256(pim.invfilter.numpy().astype(np.float64).tobytes()).hexdigest()
assert sweep_hash == preparation['sweep_sha256']
assert inverse_hash == preparation['inverse_sha256']
for name, expected in preparation['source_index_sha256'].items():
    assert digest(data / name) == expected, name
def read_index(path):
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    result = dict(rows)
    assert len(result) == len(rows) == 3000
    return result
wavs = read_index(data / 'wav.scp')
refs = [read_index(data / f'rir_ref{i}.scp') for i in (1, 2)]
assert all(set(ref) == set(wavs) for ref in refs)
prediction = output / 'prediction'
prediction.mkdir()
shutil.copy2(model / 'config.yaml', prediction / 'config.yaml')
shutil.copy2(checkpoint, prediction / 'selected_model.pth')
shutil.copy2(data / 'wav.scp', prediction / 'wav.scp')
(output / 'physical_reference.scp').write_text(''.join(
    f'{uid} {refs[0][uid]} {refs[1][uid]}\n' for uid in wavs))
metadata = dict(inference='ordinary sweep PIM', sample_rate=8000, rir_length=16000,
    count=3000, input='full utterance, channel 0, canonical FLOAT NF mixture',
    experiment=str(model), checkpoint=checkpoint.name, variant=variant,
    checkpoint_sha256=digest(checkpoint), config_sha256=digest(model / 'config.yaml'),
    input_index_sha256=digest(prediction / 'wav.scp'), output_subtype='FLOAT',
    sweep_sha256=sweep_hash, inverse_sha256=inverse_hash,
    align='peak', scale_mode='peak', pit_metric='rmse', polarity_align=False)
(prediction / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
(output / 'started.json').write_text(json.dumps(dict(
    started_at=datetime.datetime.now().astimezone().isoformat(), host=socket.gethostname(),
    gpu_index=gpu, gpu=selected[2], gpu_uuid=selected[1], model_directory=str(model),
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    scoring_commit=subprocess.check_output(['git', '-C', str(scoring_root), 'rev-parse', 'HEAD'], text=True).strip(),
    reference_preparation_sha256=digest(reference / 'preparation.json')),
    indent=2) + '\n')
PY
phase=inference
"$python" -B -m espnet2.bin.rec_rir_pit_inference \
    --train_config "$output/prediction/config.yaml" \
    --model_file "$output/prediction/selected_model.pth" \
    --wav_scp "$output/prediction/wav.scp" --output_dir "$output/prediction/inference" \
    --sample_rate 8000 --rir_length 16000 --output_subtype FLOAT --device cuda:0
phase=whamr_scoring
PYTHONPATH="$scoring_root" "$python" -B \
    "$scoring_root/egs2/whamr/rir_2spk/local/score_rir_pit.py" \
    --ref_rir_scp "$output/physical_reference.scp" \
    --pred_rir_scp "$output/prediction/inference/rir.scp" \
    --out_dir "$output/whamr" --sample_rate 8000 --rir_length 16000 \
    --align peak --scale_mode peak --pit_metric rmse
phase=ls_scoring
PYTHONPATH="$scoring_root" "$python" -B \
    "$scoring_root/egs2/whamr/rir_2spk/local/evaluate_ls_direct_reverb.py" \
    --reference_dir "$reference" \
    --prediction "$prediction_label" "$output/prediction/inference/rir.scp" \
    --out_dir "$output/ls" --no_comparison_csv
phase=finalization
"$python" -B - "$output" <<'PY'
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

output = Path(sys.argv[1])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
summaries = {}
for name, folder in [('whamr', output / 'whamr'), ('ls', output / 'ls/models/00')]:
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['num_utts'] == 3000 and summary['num_source_pairs'] == 6000
    with (folder / 'per_source.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6000
    counts = {key: sum(math.isfinite(float(row[key])) for row in rows)
              for key in ('rmse', 'rmse_direct_50ms', 'corr', 'rt60_err', 'drr_err', 'c50_err')}
    assert all(counts[key] == 6000 for key in ('rmse', 'rmse_direct_50ms', 'corr'))
    summaries[name] = dict(metrics=summary, valid_counts=counts,
                           summary_sha256=digest(folder / 'summary.json'))
assert not (output / 'ls/comparison.csv').exists()
assert len((output / 'prediction/inference/rir.scp').read_text().splitlines()) == 3000
report = dict(finished_at=datetime.datetime.now().astimezone().isoformat(),
              count=3000, source_pairs=6000, results=summaries,
              prediction_index_sha256=digest(output / 'prediction/inference/rir.scp'),
              reference_preparation_sha256=json.loads((output / 'ls/comparison.json').read_text())['reference_preparation_sha256'])
(output / 'evaluation_complete.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
phase=complete
