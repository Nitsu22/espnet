#!/usr/bin/env bash
# Submit this versioned script from the original, immutable enh_rir recipe.
# Resources and h_rt are explicitly selected by the submitter for each mode.
#$ -cwd
#$ -p -5
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
mode=${1:?train, infer, or score}
previous_status=${2:-}
cd "${SGE_O_WORKDIR:?}"
expdir=exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0
pipeline=exp/bimamba_nf8k_pipeline
logs=${pipeline}/job_${JOB_ID}
mkdir -p "${logs}"
trap 'code=$?; printf "%s\n" "$code" > "${logs}/exit_status"; date -Is' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/nf8_pipeline_numba_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nf8_pipeline_mpl_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/nf8_pipeline_triton_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}"
date -Is
git rev-parse HEAD
case ${mode} in
    train)
        if [[ -s ${pipeline}/training_complete ]]; then
            echo 'Training already completed; skip continuation'
            exit 0
        fi
        test -s "${previous_status}"
        previous=$(<"${previous_status}")
        [[ ${previous} == 0 || ${previous} == 124 ]] || exit 1
        test -s "${expdir}/checkpoint.pth"
        if [[ ${previous} == 124 ]]; then
            set +e
            bash qsub/train_nf8k_bimamba_ctf_film_all_s_1gpu.sh 86400 true
            result=$?
            set -e
            [[ ${result} == 0 ]] || exit "${result}"
        fi
        # Exit zero alone is insufficient: confirm the trainer's terminal state.
        grep -Eq '\[Early stopping\]|INFO: 150epoch results:' "${expdir}/train.log"
        test -s "${expdir}/valid.loss.best.pth"
        printf '%s\n' "${JOB_ID}" > "${pipeline}/training_complete"
        ;;
    infer)
        test -s "${pipeline}/training_complete"
        module load cuda/11.8.0
        bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 7 --stop_stage 7 \
            --ngpu 1 --python "${CONDA_PREFIX}/bin/python"
        python - <<'PY'
from pathlib import Path
exp = Path('exp/enh_train_tflocoformer_s_nf8k_bimamba_ctf_film_all_seed0')
for split in ('cv', 'tt'):
    source = Path(f'exp/bimamba_nf8k_baseline_inputs/raw/{split}_mix_clean_reverb_min_8k/wav.scp')
    def keys(path):
        rows = [line.split(maxsplit=1)[0] for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == len(set(rows)), path
        return set(rows)
    expected = keys(source)
    for speaker in (1, 2):
        scp = exp / f'inference_{split}/spk{speaker}.scp'
        assert keys(scp) == expected, scp
        for line in scp.read_text().splitlines():
            assert Path(line.split(maxsplit=1)[1]).is_file(), line
PY
        printf '%s\n' "${JOB_ID}" > "${pipeline}/inference_complete"
        ;;
    score)
        test -s "${pipeline}/inference_complete"
        bash run_nf8k_bimamba_ctf_film_all_s.sh --stage 8 --stop_stage 8 \
            --ngpu 0 --python "${CONDA_PREFIX}/bin/python"
        printf '%s\n' "${JOB_ID}" > "${pipeline}/scoring_complete"
        ;;
    *) exit 2 ;;
esac
