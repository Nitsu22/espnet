#!/usr/bin/env bash
# Submit from rir_2spk with the measured h_rt override, variant and profile dir.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -p -5
#$ -N bimamba_ablation
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
variant=${1:?Specify the ablation variant}
profile_dir=${2:?Specify the successful GPU profile directory}
resume=${3:-false}
wall_seconds=${4:?Specify the same walltime in seconds as the qsub h_rt}
case "${variant}" in ffn_only|freq_before_pool|2blocks|bilstm) ;; *) exit 2 ;; esac
case "${resume}" in true|false) ;; *) exit 2 ;; esac
[[ ${wall_seconds} =~ ^[0-9]+$ && ${wall_seconds} -ge 600 && ${wall_seconds} -le 86400 ]]
recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
cd "${recipe}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${recipe}/../../..:${recipe}/../damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_train_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_train_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bimamba_train_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
logs="exp/pooled_bimamba_ablation_train_jobs/job_${JOB_ID}"
mkdir -p "${logs}"
trap 'code=$?; echo "$code" > "$logs/exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
python - "${variant}" "${profile_dir}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

variant, profile_dir = sys.argv[1:]
config = Path(f'conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2_{variant}.yaml')
report = json.loads((Path(profile_dir) / f'{variant}.json').read_text())
assert report['config_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
assert report['batch_size'] == 4 and report['steady_update_seconds'] > 0
assert report['teacher_permutation_invariant'] and report['noise_free_mixture_verified']
assert report['rir_shape'] == [2, 32000]
assert len(report['validation_profile']) == 6
print(json.dumps(report, indent=2))
PY
tag="train_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}"
stats="exp/rir_stats_${tag}"
test -s "${stats}/train/speech_mix_shape"
test -s "${stats}/valid/speech_mix_shape"
python local/prepare_input_shapes.py --config "conf/tuning/${tag}.yaml" \
    --dump dump_nf_2spk_16k_min --output "${stats}"
# Return a recognizable status before the scheduler's hard deadline, so the
# external launcher can resume only time-limited jobs after they leave qstat.
set +e
timeout --signal=TERM --kill-after=30s "$((wall_seconds - 120))s" \
    bash "run_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.sh" \
    --stage 6 --stop_stage 6 --ngpu 1 --nj 4 \
    --python "${CONDA_PREFIX}/bin/python" --resume "${resume}"
training_status=$?
set -e
if [[ ${training_status} -eq 124 && -s exp/rir_${tag}/checkpoint.pth ]]; then
    echo 'Time limit reached; resume the existing checkpoint' > "${logs}/resume_ready"
fi
exit "${training_status}"
