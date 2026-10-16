#!/usr/bin/env bash
# Submit from enh_rir; select ONE resource explicitly on qsub's command line:
# qsub -g tga-shinoda -l gpu_1=1 -l h_rt=01:00:00 qsub/eval_same_speaker_test.sh infer TEST BASE FILM OUT
# qsub -g tga-shinoda -l cpu_4=1 -l h_rt=02:00:00 -hold_jid ID qsub/eval_same_speaker_test.sh score TEST BASE FILM OUT
# Add 'true' as sixth argument to explicitly resume a previous inference run.
# For score mode, optional arguments 7/8 are the original two score directories;
# they enable paired same-speaker/original analysis in the same CPU allocation.
#$ -cwd
#$ -p -5
#$ -N same_spk_eval
#$ -o qsub_logs
#$ -e qsub_logs
set -eo pipefail
[[ $# -eq 5 || $# -eq 6 || $# -eq 8 ]] || {
    echo 'Usage: infer|score TEST BASE FILM OUT [RESUME [ORIGINAL_BASE_SCORE ORIGINAL_FILM_SCORE]]' >&2
    exit 2
}
mode=$1; test_dir=$2; baseline_exp=$3; conditioned_exp=$4; output_dir=$5
resume=${6:-false}
case ${mode} in
    infer) [[ $# -le 6 ]] || exit 2; stage=7 ;;
    score) stage=8; resume=true ;;
    *) echo 'Use infer or score' >&2; exit 2 ;;
esac
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
logs="${PWD}/exp/same_speaker_eval_jobs/job_${JOB_ID:?}"
mkdir -p "${logs}"
export TMPDIR="${logs}/tmp"
export NUMBA_CACHE_DIR="${TMPDIR}/numba" TRITON_CACHE_DIR="${TMPDIR}/triton"
export MPLCONFIGDIR="${TMPDIR}/matplotlib" XDG_CACHE_HOME="${TMPDIR}/xdg"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TMPDIR}"
finish() {
    status=$?
    printf '%s\n' "${status}" > "${logs}/exit_status"
    # Only this job's validated, private cache/temp directory is removed.
    if [[ ${TMPDIR} == "${logs}/tmp" && -d ${TMPDIR} && ! -L ${TMPDIR} ]]; then
        rm -r -- "${TMPDIR}"
    fi
    date -Is
    exit "${status}"
}
trap finish EXIT
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
if [[ ${mode} == infer ]]; then
    module load cuda/11.8.0
    # TSUBAME exposes allocated devices without necessarily setting
    # CUDA_VISIBLE_DEVICES. Check actual CUDA availability; do not override
    # the scheduler's device selection or assume a physical GPU index.
    "${CONDA_PREFIX}/bin/python" -B - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Inference requires a usable allocated CUDA device")
print("Allocated CUDA:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
fi
set -u
shared=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..:${shared}/egs2/whamr/damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
git rev-parse HEAD > "${logs}/commit.txt"
date -Is
hostname
bash local/evaluate_same_speaker_test.sh \
    --test_dir "${test_dir}" --baseline_exp "${baseline_exp}" \
    --conditioned_exp "${conditioned_exp}" --output_dir "${output_dir}" \
    --stage "${stage}" --stop_stage "${stage}" --resume "${resume}" \
    --python "${CONDA_PREFIX}/bin/python"
if [[ ${mode} == score && $# -eq 8 ]]; then
    "${CONDA_PREFIX}/bin/python" local/analyze_same_speaker_test.py \
        --baseline "${output_dir}/baseline/score" \
        --bimamba "${output_dir}/conditioned/score" \
        --original-baseline "$7" --original-bimamba "$8" \
        --test-dir "${test_dir}" --metadata whamr_scripts/data/reverb_params_tt.csv \
        --output "${output_dir}/analysis_job_${JOB_ID}" --seed 0 --bootstrap 5000
fi
