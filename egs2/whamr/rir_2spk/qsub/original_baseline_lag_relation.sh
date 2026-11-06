#!/usr/bin/env bash
# Submit from rir_2spk after the exact original input package is transferred.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -eo pipefail
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -u
phase=${1:?Specify verify, profile or train}
variant=${2:-all}
wall_seconds=${3:-86400}
resume=${4:-false}
case ${phase} in verify|profile|train) ;; *) exit 2 ;; esac
case ${variant} in all|nb|full) ;; *) exit 2 ;; esac
case ${resume} in true|false) ;; *) exit 2 ;; esac
[[ ${wall_seconds} =~ ^[0-9]+$ && ${wall_seconds} -ge 600 && ${wall_seconds} -le 86400 ]]
cd "${SGE_O_WORKDIR:?}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*/egs2/whamr/rir_2spk) ;; *) exit 2 ;; esac
[[ ${JOB_ID:?} =~ ^[0-9]+$ ]]
repo=$(cd ../../.. && pwd)
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${repo}:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
base=exp/original_baseline_comparison
inputs=exp/original_baseline_inputs
profile=${base}/profile
logs="${base}/jobs/${phase}_${variant}_${JOB_ID}"
mkdir -p "${logs}"
task_tmp=$(mktemp -d "${PWD}/${logs}/tmp.XXXXXX")
export TMPDIR=${task_tmp} XDG_CACHE_HOME=${task_tmp}/cache
export NUMBA_CACHE_DIR=${task_tmp}/numba TRITON_CACHE_DIR=${task_tmp}/triton
export MPLCONFIGDIR=${task_tmp}/matplotlib
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    date -Is > "${logs}/ended_at"
    rm -rf -- "${task_tmp}"
    exit "${code}"
}
trap cleanup EXIT
date -Is | tee "${logs}/started_at"
hostname
git rev-parse HEAD | tee "${logs}/commit"
if [[ ${phase} == verify ]]; then
    python local/audit_original_baseline_comparison.py --inputs "${inputs}" \
        --output "${base}/input_audit.json" --workers 8 2>&1 | tee "${logs}/verify.log"
    next_job=$(qsub -g tga-shinoda -terse -p -5 -l gpu_1=1 -l h_rt=00:20:00 \
        -N recrir_orig_profile -hold_jid "${JOB_ID}" \
        qsub/original_baseline_lag_relation.sh profile all 1200 false)
    [[ ${next_job} =~ ^[0-9]+$ ]]
    printf 'profile\t%s\tgpu_1=1\t1200\n' "${next_job}" \
        | tee -a "${base}/submitted_jobs.tsv"
    printf 'GPU check submitted: %s; upper points 0.0533 (priority -5, gpu_1=1, 20min)\n' "${next_job}"
    exit 0
fi
test -s "${base}/input_audit.json"
module load cuda/11.8.0
if [[ ${phase} == profile ]]; then
    python local/profile_original_baseline_comparison.py --inputs "${inputs}" --output "${profile}" \
        2>&1 | tee "${logs}/profile.log"
    for head in nb full; do
        submission="${profile}/submitted_${head}.job"
        [[ ! -e ${submission} ]]
        next_job=$(qsub -g tga-shinoda -terse -p -5 -l gpu_1=1 -l h_rt=24:00:00 \
            -N "recrir_orig_${head}" -hold_jid "${JOB_ID}" \
            qsub/original_baseline_lag_relation.sh train "${head}" 86400 false)
        [[ ${next_job} =~ ^[0-9]+$ ]]
        printf '%s\n' "${next_job}" > "${submission}"
        printf '%s\t%s\tgpu_1=1\t86400\n' "${head}" "${next_job}" \
            | tee -a "${base}/submitted_jobs.tsv"
        printf 'Training submitted: %s %s; upper points 3.8400 (priority -5, gpu_1=1, 24h)\n' "${head}" "${next_job}"
    done
    exit 0
fi
[[ ${variant} != all ]]
config="conf/tuning/train_rec_rir_original_baseline_${variant}.yaml"
experiment="exp/rir_train_rec_rir_original_baseline_${variant}_seed0"
python - "${variant}" "${config}" "${profile}" "${inputs}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

variant, config, profile, inputs = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = json.loads((Path(profile) / 'ready.json').read_text())
assert report['commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert report['preparation_sha256'] == digest(Path(inputs) / 'preparation.json')
assert report['variants'][variant]['config_sha256'] == digest(config)
audit = json.loads(Path('exp/original_baseline_comparison/input_audit.json').read_text())
assert audit['exact_audio_bytes_verified'] is True
assert audit['preparation_sha256'] == report['preparation_sha256']
assert not report['original_baseline_retrained']
assert not report['shared_speech_initializer']
print(json.dumps(report['variants'][variant], indent=2))
PY
if [[ ${resume} == true ]]; then
    test -s "${experiment}/checkpoint.pth"
else
    [[ ! -e ${experiment}/checkpoint.pth ]]
fi
mkdir -p "${experiment}"
set +e
timeout --signal=TERM --kill-after=45s "$((wall_seconds - 120))s" \
    python -m espnet2.bin.rir_train --config "${config}" \
    --output_dir "${experiment}" --ngpu 1 --resume "${resume}" \
    2>&1 | tee "${experiment}/train.job_${JOB_ID}.log"
training_status=${PIPESTATUS[0]}
set -e
if [[ ${training_status} -eq 124 && -s ${experiment}/checkpoint.pth ]]; then
    next_job=$(qsub -g tga-shinoda -terse -p -5 -l gpu_1=1 -l "h_rt=${wall_seconds}" \
        -N "recrir_orig_${variant}" -hold_jid "${JOB_ID}" \
        qsub/original_baseline_lag_relation.sh train "${variant}" "${wall_seconds}" true)
    [[ ${next_job} =~ ^[0-9]+$ ]]
    printf '%s\n' "${next_job}" > "${logs}/continuation_job"
    printf '%s\t%s\tgpu_1=1\t%s\n' "${variant}" "${next_job}" "${wall_seconds}" \
        | tee -a "${base}/submitted_jobs.tsv"
fi
exit "${training_status}"
