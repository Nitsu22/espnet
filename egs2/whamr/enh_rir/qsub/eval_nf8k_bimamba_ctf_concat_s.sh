#!/usr/bin/env bash
# Select resources/h_rt at submission: infer on one GPU, score on CPU.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
mode=${1:?infer or score}
case ${mode} in infer|score) ;; *) exit 2 ;; esac
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*) ;; *) exit 2 ;; esac
[[ ${JOB_ID:?} =~ ^[0-9]+$ ]]
expdir=exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0
pipeline=exp/bimamba_nf8k_concat_eval_normalized
export CONCAT_INFERENCE_TAG=inference_normalized
logs=${PWD}/${pipeline}/job_${JOB_ID}
mkdir -p "${logs}"
task_tmp=$(mktemp -d "${logs}/tmp.XXXXXX")
export TMPDIR=${task_tmp} XDG_CACHE_HOME=${task_tmp}/cache
export NUMBA_CACHE_DIR=${task_tmp}/numba TRITON_CACHE_DIR=${task_tmp}/triton
export MPLCONFIGDIR=${task_tmp}/matplotlib PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    case ${task_tmp} in "${logs}"/tmp.*) rm -rf -- "${task_tmp}" ;; esac
    date -Is
    exit "${code}"
}
trap cleanup EXIT
date -Is
git rev-parse HEAD | tee "${logs}/commit.txt"
test -s "${expdir}/valid.loss.best.pth"
grep -Eq '\[Early stopping\]|INFO: 150epoch results:' "${expdir}/train.log"
if [[ ${mode} == infer ]]; then
    module load cuda/11.8.0
    # Never overwrite the previous, unnormalized inference/evaluation.
    for split in cv tt; do
        test ! -e "${expdir}/${CONCAT_INFERENCE_TAG}_${split}"
    done
    bash run_nf8k_bimamba_ctf_concat_s.sh --stage 7 --stop_stage 7 \
        --ngpu 1 --inference_model valid.loss.best.pth \
        --normalize_output_wav true --inference_tag "${CONCAT_INFERENCE_TAG}" \
        --python "${CONDA_PREFIX}/bin/python"
else
    previous_job=${2:?inference job ID}
    [[ ${previous_job} =~ ^[0-9]+$ ]]
    test "$(<"${pipeline}/job_${previous_job}/exit_status")" = 0
    test "$(<"${pipeline}/inference_complete")" = "${previous_job}"
fi
# Both stages reject missing/duplicate UIDs and incomplete inference output.
python - <<'PY'
from pathlib import Path
import os
import numpy as np
import soundfile as sf

exp = Path('exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_concat_seed0')
tag = os.environ['CONCAT_INFERENCE_TAG']


def rows(path):
    pairs = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    assert all(len(pair) == 2 for pair in pairs), path
    mapping = dict(pairs)
    assert len(pairs) == len(mapping), f'Duplicate UID: {path}'
    return mapping


for split, count in (('cv', 5000), ('tt', 3000)):
    source = Path(f'exp/bimamba_nf8k_baseline_inputs/raw/{split}_mix_clean_reverb_min_8k/wav.scp')
    expected = rows(source)
    assert len(expected) == count, source
    for speaker in (1, 2):
        scp = exp / f'{tag}_{split}/spk{speaker}.scp'
        output = rows(scp)
        assert output.keys() == expected.keys(), scp
        assert all(Path(path).is_file() for path in output.values()), scp
        # Check a deterministic sample before accepting output as normalized.
        for uid in sorted(output)[::max(1, count // 100)]:
            audio, rate = sf.read(output[uid])
            assert rate == 8000 and np.isfinite(audio).all(), uid
            assert np.max(np.abs(audio)) <= 0.901, f'Clipping/unnormalized: {uid}'
PY
if [[ ${mode} == infer ]]; then
    printf '%s\n' "${JOB_ID}" > "${pipeline}/inference_complete"
else
    bash run_nf8k_bimamba_ctf_concat_s.sh --stage 8 --stop_stage 8 \
        --ngpu 0 --inference_model valid.loss.best.pth \
        --normalize_output_wav true --inference_tag "${CONCAT_INFERENCE_TAG}" \
        --python "${CONDA_PREFIX}/bin/python"
    for split in cv tt; do
        for speaker in 1 2; do
            for metric in SI_SNR STOI ESTOI SDR SAR SIR; do
                test -s "${expdir}/${CONCAT_INFERENCE_TAG}_${split}/score/${metric}_spk${speaker}"
            done
        done
    done
    printf '%s\n' "${JOB_ID}" > "${pipeline}/scoring_complete"
fi
