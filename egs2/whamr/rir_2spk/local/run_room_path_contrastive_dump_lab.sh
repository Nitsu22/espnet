#!/usr/bin/env bash
# Run on the CPU of the chosen lab host. Preserve logs; clean task scratch on exit.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: script FINAL_DUMP RESULT_DIR EXPECTED_COMMIT' >&2; exit 2; }
recipe=$(cd "$(dirname "$0")/.." && pwd)
code_root=$(cd "${recipe}/../../.." && pwd)
final_dump=$(realpath -m "$1")
result_dir=$(realpath -m "$2")
expected_commit=$3
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
source_recipe=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk
source_dump="${source_recipe}/dump_nf_2spk_16k_min"
metadata_root=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/enh_rir/data_rir_plus/wav16k/min
case ${final_dump} in /net/midgar/work2/nitsu/*) ;; *) echo 'Unexpected dump destination' >&2; exit 2 ;; esac
case ${result_dir} in /net/midgar/work2/nitsu/*) ;; *) echo 'Unexpected result destination' >&2; exit 2 ;; esac
[[ ! -e ${final_dump} ]] || { echo 'Final dump already exists' >&2; exit 1; }
mkdir -p "${result_dir}"
[[ ! -e ${result_dir}/pipeline_status.json ]] || { echo 'Use a fresh result directory' >&2; exit 1; }
[[ $(git -C "${code_root}" rev-parse HEAD) == "${expected_commit}" ]] || { echo 'Code commit differs' >&2; exit 1; }
[[ -z $(git -C "${code_root}" status --porcelain) ]] || { echo 'Code checkout has uncommitted changes' >&2; exit 1; }
[[ $(git -C "${code_root}" branch --show-current) == codex/room-path-contrastive-20261013 ]] || exit 1
git -C "${code_root}" pull --ff-only
[[ $(git -C "${code_root}" rev-parse HEAD) == "${expected_commit}" ]] || exit 1
scratch_root=/tmp/nitsu
if [[ ! -e ${scratch_root} ]]; then mkdir -m 700 "${scratch_root}"; fi
[[ -d ${scratch_root} && ! -L ${scratch_root} && $(stat -c %u "${scratch_root}") == $(id -u) ]] || {
    echo 'Scratch root is not an owned directory' >&2; exit 1;
}
task_tmp=$(mktemp -d "${scratch_root}/room-path-paired.XXXXXXXX")
export TMPDIR="${task_tmp}"
export NUMBA_CACHE_DIR="${task_tmp}/numba"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
staged_dump="${task_tmp}/dump"
publish_dir="${final_dump}.partial-$(basename "${task_tmp}")"
phase=initializing
cleanup() {
    status=$?
    trap - EXIT INT TERM
    set +e
    # Remove only the exact private directories created by this invocation.
    if [[ -d ${publish_dir} ]]; then rm -rf -- "${publish_dir}"; fi
    if [[ -d ${task_tmp} && ! -L ${task_tmp} && $(stat -c %u "${task_tmp}") == $(id -u) ]]; then
        rm -rf -- "${task_tmp}"
    fi
    if [[ -e ${task_tmp} ]]; then status=1; fi
    "${python}" - "${result_dir}" "${status}" "${phase}" "${task_tmp}" "${expected_commit}" "${final_dump}" <<'PY'
import json, os, socket, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
dest, status, phase, scratch, commit, dump = sys.argv[1:]
report = dict(exit_status=int(status), phase=phase, host=socket.gethostname(),
              gpu=None, scratch_directory=scratch, scratch_removed=not os.path.exists(scratch),
              git_commit=commit, dump=dump,
              finished=datetime.now(timezone(timedelta(hours=9))).isoformat())
Path(dest, 'pipeline_status.json').write_text(json.dumps(report, indent=2)+'\n')
Path(dest, 'exit_status').write_text(status+'\n')
PY
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${python}" - "${result_dir}" "${task_tmp}" "${expected_commit}" <<'PY'
import json, socket, sys
from pathlib import Path
Path(sys.argv[1], 'launch.json').write_text(json.dumps(dict(
    host=socket.gethostname(), cpu_workers=4, scratch=sys.argv[2], commit=sys.argv[3]), indent=2)+'\n')
PY
cd "${task_tmp}"
generator="${recipe}/local/prepare_room_path_contrastive_dump.py"
phase=capacity_check
"${python}" "${generator}" --source-dump "${source_dump}" --metadata-root "${metadata_root}" \
    --output-dir "${staged_dump}" --index-root "${final_dump}" --estimate-only > "${result_dir}/capacity.json"
"${python}" - "${result_dir}/capacity.json" "${task_tmp}" "$(dirname "${final_dump}")" <<'PY'
import json, shutil, sys
report=json.loads(open(sys.argv[1]).read())['estimate']
needed=report['recommended_free_bytes']
for directory in sys.argv[2:]:
    available=shutil.disk_usage(directory).free
    if available < needed + 20 * 2**30:
        raise RuntimeError('Insufficient capacity: '+directory)
print('Capacity verified', report['estimated_total_gib'], 'GiB')
PY
phase=generating
"${python}" "${generator}" --source-dump "${source_dump}" --metadata-root "${metadata_root}" \
    --output-dir "${task_tmp}/smoke" --workers 2 --seed 20261013 --limit 2
cp "${task_tmp}/smoke/verification.json" "${result_dir}/smoke_verification.json"
"${python}" "${generator}" --source-dump "${source_dump}" --metadata-root "${metadata_root}" \
    --output-dir "${staged_dump}" --index-root "${final_dump}" --workers 4 --seed 20261013
phase=publishing
mkdir "${publish_dir}"
rsync -a "${staged_dump}/" "${publish_dir}/"
"${python}" "${generator}" --verify-existing "${publish_dir}"
[[ ! -e ${final_dump} ]] || { echo 'Destination appeared during generation' >&2; exit 1; }
mv "${publish_dir}" "${final_dump}"
phase=recording_final
cp "${final_dump}/verification.json" "${result_dir}/verification.json"
cp "${final_dump}/completion.json" "${result_dir}/generation_complete.json"
phase=complete
