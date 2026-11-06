#!/usr/bin/env bash
# Resources and h_rt are supplied by the submission command, from rir_2spk.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -eo pipefail
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -u
phase=${1:?Specify prepare, profile or train}
variant=${2:-all}
wall_seconds=${3:-86400}
resume=${4:-false}
submit_after_profile=${5:-false}
case ${phase} in prepare|profile|train) ;; *) exit 2 ;; esac
case ${variant} in all|baseline|nb|full) ;; *) exit 2 ;; esac
case ${resume} in true|false) ;; *) exit 2 ;; esac
case ${submit_after_profile} in true|false) ;; *) exit 2 ;; esac
[[ ${wall_seconds} =~ ^[0-9]+$ && ${wall_seconds} -ge 600 && ${wall_seconds} -le 86400 ]]
cd "${SGE_O_WORKDIR:?}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*/egs2/whamr/rir_2spk) ;; *) exit 2 ;; esac
[[ ${JOB_ID:?} =~ ^[0-9]+$ ]]
repo=$(cd ../../.. && pwd)
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${repo}:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
logs="exp/lag_relation_comparison/jobs/${phase}_${variant}_${JOB_ID}"
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
dump=dump_nf8k_fair
profile=exp/lag_relation_comparison/profile
stats=exp/rir_stats_nf8k_fair
if [[ ${phase} == prepare ]]; then
    python local/prepare_lag_relation_comparison.py \
        --source "${shared}/egs2/whamr/rir_2spk/dump_nf_2spk_16k_min" \
        --output "${PWD}/${dump}" --workers 8 2>&1 | tee "${logs}/prepare.log"
    python local/prepare_input_shapes.py \
        --config conf/tuning/train_rec_rir_nf8k_fair_baseline.yaml \
        --dump "${dump}" --output "${stats}"
    exit 0
fi
test -s "${dump}/preparation.json"
test -s "${stats}/train/speech_mix_shape"
test -s "${stats}/valid/speech_mix_shape"
module load cuda/11.8.0
if [[ ${phase} == profile ]]; then
    python local/profile_lag_relation_comparison.py --dump "${dump}" --output "${profile}" \
        2>&1 | tee "${logs}/profile.log"
    # Submit GPU training only after the complete three-model gate succeeds.
    # A failed gate should not consume three GPU allocations just to fail on
    # a missing ready.json. Persist each job ID immediately for recovery.
    if [[ ${submit_after_profile} == true ]]; then
        for head in baseline nb full; do
            submission="${profile}/submitted_${head}.job"
            [[ ! -e ${submission} ]]
            train_job=$(qsub -g tga-shinoda -terse -p -5 -l gpu_1=1 \
                -l h_rt=24:00:00 -N "recrir_fair_${head}" -hold_jid "${JOB_ID}" \
                qsub/lag_relation_comparison.sh train "${head}" 86400 false)
            [[ ${train_job} =~ ^[0-9]+$ ]]
            printf '%s\n' "${train_job}" > "${submission}"
            printf '%s\t%s\tgpu_1=1\t86400\n' "${head}" "${train_job}" \
                | tee -a exp/lag_relation_comparison/submitted_jobs.tsv
            printf 'Submitted %s: %s; priority -5, gpu_1=1, h_rt=24h, upper points 3.8400\n' \
                "${head}" "${train_job}"
        done
    fi
    exit 0
fi
[[ ${variant} != all ]]
config="conf/tuning/train_rec_rir_nf8k_fair_${variant}.yaml"
experiment="exp/rir_train_rec_rir_nf8k_fair_${variant}_seed0"
python - "${variant}" "${config}" "${profile}" "${dump}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

variant, config, profile, dump = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = json.loads((Path(profile) / 'ready.json').read_text())
assert report['commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert report['dataset_sha256'] == digest(Path(dump) / 'preparation.json')
assert report['shared_init_sha256'] == digest(Path(profile) / 'shared_speech_init.pth')
assert report['variants'][variant]['config_sha256'] == digest(config)
assert report['variants'][variant]['batch_size'] == 4
print(json.dumps(report['variants'][variant], indent=2))
PY
if [[ ${resume} == true ]]; then
    test -s "${experiment}/checkpoint.pth"
else
    [[ ! -e ${experiment}/checkpoint.pth ]]
fi
mkdir -p "${experiment}"
data_args=()
for name in speech_mix speech_direct1 speech_direct2 speech_reverb1 speech_reverb2; do
    data_args+=(--train_data_path_and_name_and_type "${dump}/raw/tr_rir_2spk_nf_min_8k/${name}.scp,${name},sound")
    data_args+=(--valid_data_path_and_name_and_type "${dump}/raw/cv_rir_2spk_nf_min_8k/${name}.scp,${name},sound")
done
set +e
timeout --signal=TERM --kill-after=45s "$((wall_seconds - 120))s" \
    python -m espnet2.bin.rir_train --config "${config}" \
    --output_dir "${experiment}" --ngpu 1 --resume "${resume}" \
    --init_param "${profile}/shared_speech_init.pth" \
    --train_shape_file "${stats}/train/speech_mix_shape" \
    --valid_shape_file "${stats}/valid/speech_mix_shape" --fold_length 80000 \
    "${data_args[@]}" 2>&1 | tee "${experiment}/train.job_${JOB_ID}.log"
training_status=${PIPESTATUS[0]}
set -e
# Resume only a deliberate time limit with a saved checkpoint. Other failures stop.
if [[ ${training_status} -eq 124 && -s ${experiment}/checkpoint.pth ]]; then
    continuation=$(qsub -g tga-shinoda -terse -p -5 -l gpu_1=1 \
        -l "h_rt=${wall_seconds}" -N "recrir_fair_${variant}" -hold_jid "${JOB_ID}" \
        qsub/lag_relation_comparison.sh train "${variant}" "${wall_seconds}" true)
    printf '%s\n' "${continuation}" > "${logs}/continuation_job"
fi
exit "${training_status}"
