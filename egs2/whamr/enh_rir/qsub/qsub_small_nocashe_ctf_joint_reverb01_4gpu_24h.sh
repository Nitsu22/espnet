#!/usr/bin/env bash
# Submit from egs2/whamr/enh_rir with qsub -g tga-shinoda.
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N ctf_s4_r01_24h
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
if __conda_setup="$(/gs/bs/tga-shinoda/nitsu/anaconda3/bin/conda shell.bash hook 2>/dev/null)"; then
    eval "${__conda_setup}"
else
    . /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
fi
unset __conda_setup
module load cuda/11.8.0
conda activate tf-locoformer
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nitsu_ctf_numba_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

printf 'Commit: %s\n' "$(git rev-parse HEAD)"
nvidia-smi -L
python - <<'PY'
import json
from pathlib import Path

dump = Path("dump_ctf_joint")
manifest = json.loads((dump / "manifest.json").read_text())
assert manifest["sample_rate"] == 8000
assert manifest["transfer_validation"].startswith("passed")
assert {s: info["count"] for s, info in manifest["splits"].items()} == {
    "tr": 20000, "cv": 5000, "tt": 3000
}
print("Verified 8-kHz joint-CTF dump is ready", flush=True)
PY

# Change only the reverb-loss weight to 0.1; start a separate experiment.
./run_small_nocashe_ctf_joint_4gpu.sh \
    --ngpu 4 --stage 6 --stop_stage 6 \
    --resume false \
    --enh_config conf/tuning/train_enh_tflocoformer_small_nocashe_ctf_joint_reverb01_4gpu.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_small_nocashe_ctf_joint_reverb01_4gpu
