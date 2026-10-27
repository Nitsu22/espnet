#!/usr/bin/env bash
# One GPU per variant; smoke-test two mixtures before the full raw-CA test.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: GPU_INDEX RESULT_RECIPE bimamba|rec_rir|bilstm|damsep' >&2; exit 2; }
gpu=$1
result=$(realpath "$2")
variant=$3
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
case "$variant" in bimamba|rec_rir|bilstm|damsep) ;; *) exit 2 ;; esac
case "$result" in /net/midgar/work2/nitsu/*|/gs/bs/tga-shinoda/nitsu/*) ;; *) exit 1 ;; esac
[[ -O "$result" && -O "$result/exp" ]] || exit 1
recipe=$(cd "$(dirname "$0")/.." && pwd)
code=$(git -C "$recipe" rev-parse --show-toplevel)
cd "$code"
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
control="$result/exp/oracle_cd_ca_${variant}_control_20261027"
output="$result/exp/oracle_cd_ca_${variant}_eval8k_20261027"
smoke="$result/exp/oracle_cd_ca_${variant}_smoke8k_20261027"
for path in "$control" "$output" "$smoke"; do
 [[ ! -e "$path" && ! -L "$path" ]] || { echo "Existing output: $path" >&2; exit 1; }
done
mkdir "$control" "$control/.tmp"
phase=initialization
finish() {
 status=$?
 trap - EXIT
 printf '%s\n' "$status" > "$control/exit_status"
 printf '%s\n' "$phase" > "$control/final_phase"
 rm -rf -- "$control/.tmp"
 exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES="$gpu" TZ=Asia/Tokyo
export PYTHONPATH="$code" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$control/.tmp" NUMBA_CACHE_DIR="$control/.tmp/numba"
export XDG_CACHE_HOME="$control/.tmp/cache" TRITON_CACHE_DIR="$control/.tmp/triton"
export TORCH_EXTENSIONS_DIR="$control/.tmp/torch_extensions" CUDA_CACHE_PATH="$control/.tmp/cuda"
export TORCH_HOME="$control/.tmp/torch" MPLCONFIGDIR="$control/.tmp/matplotlib"
phase=gpu_validation
"$python" -B - "$gpu" "$control" <<'PY'
import csv, datetime, io, json, socket, subprocess, sys
from pathlib import Path
rows=list(csv.reader(io.StringIO(subprocess.check_output([
 'nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True))))
selected=next([s.strip() for s in r] for r in rows if int(r[0])==int(sys.argv[1]))
apps=list(csv.reader(io.StringIO(subprocess.check_output([
 'nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True))))
assert int(selected[3])<200 and not any(r and r[0].strip()==selected[1] for r in apps), 'GPU is occupied'
Path(sys.argv[2],'started.json').write_text(json.dumps(dict(
 checked_at=datetime.datetime.now().astimezone().isoformat(),host=socket.gethostname(),
 gpu_index=int(selected[0]),gpu=selected[2],gpu_uuid=selected[1]),indent=2)+'\n')
PY
phase=smoke
"$python" -B "$recipe/local/evaluate_oracle_cd_ca.py" --recipe "$result" \
 --variant "$variant" --output "$smoke" --limit 2
phase=evaluation
"$python" -B "$recipe/local/evaluate_oracle_cd_ca.py" --recipe "$result" \
 --variant "$variant" --output "$output"
phase=finalization
"$python" -B - "$output" "$control" <<'PY'
import json, sys
from pathlib import Path
output, control=map(Path,sys.argv[1:])
report=json.loads((output/'evaluation_complete.json').read_text())
assert report['count']==3000 and report['source_pairs']==6000
summary=json.loads((output/'summary.json').read_text())
assert not summary['metadata']['reference_sweep_processing']
assert summary['summary']['valid_counts']['rmse']==6000
assert summary['summary']['valid_counts']['corr']==6000
assert len((output/'rir.scp').read_text().splitlines())==3000
(control/'evaluation_complete.json').write_text(json.dumps(report,indent=2)+'\n')
PY
phase=complete
