#!/usr/bin/env bash
# Check the real paired model, then start scratch training on one idle GPU.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script GPU_INDEX EXP_DIRECTORY EXPECTED_COMMIT' >&2; exit 2; }
gpu=$1
[[ ${gpu} =~ ^[0-9]+$ ]] || exit 2
experiment=$(realpath -m "$2")
expected_commit=$3
variant=paired_only
suffix=_paired_only
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(cd "${recipe}/../../.." && pwd)
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
main_recipe=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
dump="${main_recipe}/dump_nf_2spk_16k_min_contrastive"
stats="${main_recipe}/exp/rir_stats_train_pooled_bimamba_2spk_nf_16k_room_path_contrastive"
config="${recipe}/conf/tuning/train_pooled_bimamba_2spk_nf_16k_room_path_contrastive${suffix}.yaml"
case ${experiment} in "${main_recipe}/exp/"*) ;; *) echo 'Unexpected experiment directory' >&2; exit 2 ;; esac
[[ ! -e ${experiment} && $(stat -c %u "$(dirname "${experiment}")") == $(id -u) ]] || {
    echo 'Use a fresh directory beneath the owned recipe exp directory' >&2; exit 1;
}
case $(git -C "${code_root}" branch --show-current) in
    codex/room-path-contrastive-20261013|codex/room-path-loss-ablations-20261014) ;;
    *) echo "Unexpected code branch" >&2; exit 1 ;;
esac
[[ $(git -C "${code_root}" rev-parse HEAD) == "${expected_commit}" ]] || exit 1
[[ -z $(git -C "${code_root}" status --porcelain) ]] || { echo 'Uncommitted code changes' >&2; exit 1; }
[[ -f ${dump}/completion.json && -f ${stats}/input_shapes.json ]] || exit 1
mkdir "${experiment}"
exec >"${experiment}/run.log" 2>&1
export CUDA_VISIBLE_DEVICES="${gpu}" TZ=Asia/Tokyo PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cache="${experiment}/runtime_cache"
mkdir -p "${cache}"
export NUMBA_CACHE_DIR="${cache}/numba" MPLCONFIGDIR="${cache}/matplotlib"
export XDG_CACHE_HOME="${cache}/cache" TORCH_EXTENSIONS_DIR="${cache}/torch_extensions"
export CUDA_CACHE_PATH="${cache}/cuda" TRITON_CACHE_DIR="${cache}/triton"
task_tmp=$(mktemp -d "${experiment}/runtime_tmp.XXXXXXXX")
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
    "${python}" - "${experiment}" "${status}" "${phase}" "${task_tmp}" <<'PY'
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
import csv, subprocess, sys, time
gpu=sys.argv[1]
for attempt in range(6):
    row=next(csv.reader(subprocess.check_output([
        'nvidia-smi', '-i', gpu, '--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()))
    apps=list(csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()))
    if any(app[0].strip()==row[0].strip() for app in apps):
        raise RuntimeError('Selected GPU has a compute process; no job was stopped')
    if int(row[1])<=256 and int(row[2])<=5:
        print('Idle GPU verified:', gpu, row[0])
        break
    if attempt==5:
        raise RuntimeError('Selected GPU did not become idle')
    time.sleep(2)
PY
}
check_gpu
"${python}" - "${experiment}" "${gpu}" "${expected_commit}" "${config}" "${dump}" <<'PY'
import hashlib, json, socket, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
root, gpu, commit, config, dump=sys.argv[1:]
report=dict(host=socket.gethostname(), gpu_index=int(gpu), commit=commit,
            config=config, config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
            dump=dump, experiment=root,
            started=datetime.now(timezone(timedelta(hours=9))).isoformat())
Path(root, 'started.json').write_text(json.dumps(report, indent=2)+'\n')
PY
phase=gpu_readiness
"${python}" "${recipe}/local/check_room_path_contrastive_training.py" \
    --config "${config}" --data-dir "${dump}/raw/tr_rir_2spk_nf_min_16k" \
    --output "${experiment}/gpu_readiness.json"
check_gpu
phase=training
"${python}" - "${experiment}" <<'PY'
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
Path(sys.argv[1], 'phase.json').write_text(json.dumps(dict(
    phase='training', started=datetime.now(timezone(timedelta(hours=9))).isoformat()), indent=2)+'\n')
PY
cd "${recipe}"
bash run_pooled_bimamba_2spk_nf_16k_room_path_contrastive.sh \
    --stage 6 --stop_stage 6 --resume false --ngpu 1 \
    --rir_exp "${experiment}" --rir_stats_dir "${stats}" --rir_config "${config}"
[[ -f ${experiment}/checkpoint.pth && -f ${experiment}/valid.loss_sweep.best.pth ]] || {
    echo 'Training returned without required checkpoints' >&2; exit 1;
}
phase=complete
