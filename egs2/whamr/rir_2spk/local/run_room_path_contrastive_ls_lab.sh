#!/usr/bin/env bash
# Ordinary sweep PIM -> immutable LS direct-reverb reference, for two controls.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script GPU_INDEX both|paired_only EXPECTED_COMMIT' >&2; exit 2; }
gpu=$1
variant=$2
expected_commit=$3
[[ ${gpu} =~ ^[0-9]+$ ]] || exit 2
case ${variant} in
    both) suffix= ;;
    paired_only) suffix=_paired_only ;;
    *) echo 'Choose both or paired_only' >&2; exit 2 ;;
esac
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(cd "${recipe}/../../.." && pwd)
main_recipe=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
experiment="${main_recipe}/exp/rir_train_pooled_bimamba_2spk_nf_16k_room_path_contrastive${suffix}"
data="${main_recipe}/dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k"
reference="${main_recipe}/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_16k"
evaluation="${main_recipe}/exp/room_path_contrastive_ls_20261018/${variant}"
[[ $(stat -c %u "${main_recipe}/exp") == $(id -u) ]] || exit 1
[[ $(git -C "${code_root}" branch --show-current) == codex/room-path-loss-ablations-20261014 ]] || exit 1
[[ $(git -C "${code_root}" rev-parse HEAD) == "${expected_commit}" ]] || exit 1
[[ -z $(git -C "${code_root}" status --porcelain) ]] || exit 1
[[ ! -e ${evaluation} && -f ${experiment}/pipeline_status.json ]] || exit 1
mkdir -p "$(dirname "${evaluation}")"
mkdir "${evaluation}"
exec >"${evaluation}/run.log" 2>&1
export CUDA_VISIBLE_DEVICES="${gpu}" TZ=Asia/Tokyo PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cache="${evaluation}/runtime_cache"
mkdir -p "${cache}"
export NUMBA_CACHE_DIR="${cache}/numba" MPLCONFIGDIR="${cache}/matplotlib"
export XDG_CACHE_HOME="${cache}/cache" TORCH_EXTENSIONS_DIR="${cache}/torch_extensions"
export CUDA_CACHE_PATH="${cache}/cuda" TRITON_CACHE_DIR="${cache}/triton"
task_tmp=$(mktemp -d "${evaluation}/runtime_tmp.XXXXXXXX")
export TMPDIR="${task_tmp}"
phase=preflight
finish() {
    status=$?
    trap - EXIT INT TERM
    set +e
    if [[ -d ${task_tmp} && ! -L ${task_tmp} && $(stat -c %u "${task_tmp}") == $(id -u) ]]; then
        rm -rf -- "${task_tmp}"
    fi
    [[ ! -e ${task_tmp} ]] || status=1
    "${python}" - "${evaluation}" "${status}" "${phase}" "${task_tmp}" <<'PY'
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
root, status, phase, scratch = sys.argv[1:]
report = dict(exit_status=int(status), phase=phase, temporary_directory=scratch,
              temporary_removed=not Path(scratch).exists(),
              finished=datetime.now(timezone(timedelta(hours=9))).isoformat())
Path(root, 'pipeline_status.json').write_text(json.dumps(report, indent=2)+'\n')
Path(root, 'exit_status').write_text(status+'\n')
PY
    exit "${status}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
check_gpu() {
    "${python}" - "${gpu}" <<'PY'
import csv, subprocess, sys
gpu=sys.argv[1]
row=next(csv.reader(subprocess.check_output([
    'nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).splitlines()))
apps=list(csv.reader(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()))
if any(app[0].strip()==row[0].strip() for app in apps) or int(row[1])>256 or int(row[2])>5:
    raise RuntimeError('Selected GPU is occupied; no process was stopped')
print('Idle GPU verified:', gpu, row[0], flush=True)
PY
}
check_gpu
"${python}" - "${evaluation}" "${experiment}" "${data}" "${reference}" "${gpu}" "${expected_commit}" <<'PY'
import hashlib, json, socket, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
out, experiment, data, reference=map(Path, sys.argv[1:5])
training=json.loads((experiment/'pipeline_status.json').read_text())
if training['exit_status']!=0 or training['phase']!='complete':
    raise ValueError('Training must have completed successfully')
ids=[line.split()[0] for line in (data/'wav.scp').read_text().splitlines()]
ref_ids=[line.split()[0] for line in (reference/'reference.scp').read_text().splitlines()]
if len(ids)!=3000 or len(set(ids))!=3000 or set(ids)!=set(ref_ids) or len(ref_ids)!=3000:
    raise ValueError('Require the complete shared 3000-mixture test set')
source=json.loads((reference/'preparation.json').read_text())
if source['config']['sample_rate']!=16000 or source['config']['rir_length']!=32000:
    raise ValueError('Expected 16 kHz, two-second LS targets')
if Path(source['source_data']).resolve()!=data.resolve() or source['channel']!=0:
    raise ValueError('LS reference source/channel differs from inference data')
for name, checksum in source['source_index_sha256'].items():
    if hashlib.sha256((data/name).read_bytes()).hexdigest()!=checksum:
        raise ValueError('LS reference source index changed: '+name)
checkpoint=experiment/'valid.loss_sweep.best.pth'
report=dict(host=socket.gethostname(), gpu_index=int(sys.argv[5]), commit=sys.argv[6],
            experiment=str(experiment), data=str(data), reference=str(reference), count=3000,
            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            reference_preparation_sha256=hashlib.sha256((reference/'preparation.json').read_bytes()).hexdigest(),
            started=datetime.now(timezone(timedelta(hours=9))).isoformat())
(out/'started.json').write_text(json.dumps(report, indent=2)+'\n')
PY
phase=reference_validation
"${python}" "${recipe}/local/prepare_ls_rir_reference.py" --mode validate --output "${reference}" \
    >"${evaluation}/reference_validation.json"
check_gpu
phase=smoke_inference
"${python}" "${recipe}/local/evaluate_two_speaker.py" \
    --experiment "${experiment}" --checkpoint valid.loss_sweep.best.pth \
    --data "${data}" --output "${evaluation}/smoke" --device cuda:0 --limit 2 --stage inference
phase=smoke_scoring
"${python}" - "${evaluation}/smoke/inference/rir.scp" "${reference}/reference.scp" "${evaluation}/smoke/ls_reference.scp" <<'PY'
from pathlib import Path
import sys
predicted={line.split()[0] for line in Path(sys.argv[1]).read_text().splitlines()}
rows=[line for line in Path(sys.argv[2]).read_text().splitlines() if line.split()[0] in predicted]
if len(predicted)!=2 or len(rows)!=2:
    raise ValueError('Smoke reference/prediction IDs differ')
Path(sys.argv[3]).write_text('\n'.join(rows)+'\n')
PY
"${python}" "${recipe}/local/score_rir_pit.py" \
    --ref_rir_scp "${evaluation}/smoke/ls_reference.scp" \
    --pred_rir_scp "${evaluation}/smoke/inference/rir.scp" \
    --out_dir "${evaluation}/smoke/score_ls" --sample_rate 16000 --rir_length 32000 \
    --align peak --scale_mode peak --pit_metric rmse
check_gpu
phase=inference
"${python}" "${recipe}/local/evaluate_two_speaker.py" \
    --experiment "${experiment}" --checkpoint valid.loss_sweep.best.pth \
    --data "${data}" --output "${evaluation}/whamr" --device cuda:0 --stage inference
phase=ls_scoring
"${python}" "${recipe}/local/evaluate_ls_direct_reverb.py" \
    --reference_dir "${reference}" --prediction "${variant}" "${evaluation}/whamr/inference/rir.scp" \
    --out_dir "${evaluation}/ls"
"${python}" - "${evaluation}" "${variant}" <<'PY'
import csv, hashlib, json, math, sys
from pathlib import Path
out=Path(sys.argv[1])
report=json.loads((out/'ls/comparison.json').read_text())
started=json.loads((out/'started.json').read_text())
model=report['models'][sys.argv[2]]
if (report['count']!=3000 or report['source_pairs']!=6000
        or report['reference_preparation_sha256']!=started['reference_preparation_sha256']
        or model['prediction_metadata']['checkpoint_sha256']!=started['checkpoint_sha256']):
    raise ValueError('Count, reference, or checkpoint provenance mismatch')
with (out/'ls/models/00/per_source.csv').open() as stream:
    rows=list(csv.DictReader(stream))
if len(rows)!=6000 or any(not math.isfinite(float(r[k])) for r in rows
                         for k in ('rmse','rmse_50ms','rmse_direct_50ms','corr')):
    raise ValueError('Incomplete/nonfinite waveform scores')
marker=dict(count=3000, source_pairs=6000, model=sys.argv[2],
            checkpoint_sha256=started['checkpoint_sha256'],
            reference_preparation_sha256=started['reference_preparation_sha256'],
            comparison_sha256=hashlib.sha256((out/'ls/comparison.json').read_bytes()).hexdigest())
(out/'evaluation_complete.json').write_text(json.dumps(marker, indent=2)+'\n')
PY
phase=complete
