#!/usr/bin/env bash
# Submit only after CPU preparation and the GPU validation have succeeded.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N tfs_oracle_ctf
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh9_condition}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/oracle_ctf_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/oracle_ctf_mpl_${JOB_ID}"
resume=${1:-false}
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
p = json.loads(Path('exp/oracle_ctf_cuda_check.json').read_text())
assert p['status'] == 'passed' and p['device'] == 'cuda'
assert p['counts'] == {'tr': 20000, 'cv': 5000, 'tt': 3000}
assert p['config_sha256'] == hashlib.sha256(Path('conf/tuning/train_enh_tflocoformer_s_nf8k_oracle_ctf_concat.yaml').read_bytes()).hexdigest()
assert p['manifest_sha256'] == hashlib.sha256(Path('dump_oracle_ctf_nf8k/preparation.json').read_bytes()).hexdigest()
assert p['commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Verified CUDA readiness for this commit, config and dump')
PY
bash run_oracle_ctf_concat.sh --stage 6 --stop_stage 6 --ngpu 1 \
    --python "${CONDA_PREFIX}/bin/python" --resume "${resume}"
