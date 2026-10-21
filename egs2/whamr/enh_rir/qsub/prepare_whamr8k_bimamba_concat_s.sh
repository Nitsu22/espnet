#!/usr/bin/env bash
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -p -5
#$ -N prep_bi8_whamr
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
cd "${SGE_O_WORKDIR:?Submit from enh_rir}"
case ${PWD} in /gs/bs/tga-shinoda/nitsu/*) ;; *) exit 2 ;; esac
logs="${PWD}/exp/bimamba_whamr8k_concat_jobs/prepare_${JOB_ID:?}"
mkdir -p "${logs}"
job_tmp=$(mktemp -d "${logs}/tmp.XXXXXX")
export TMPDIR="${job_tmp}" XDG_CACHE_HOME="${job_tmp}/cache"
export NUMBA_CACHE_DIR="${job_tmp}/numba" MPLCONFIGDIR="${job_tmp}/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${logs}/exit_status"
    case ${job_tmp} in "${logs}"/tmp.*) rm -rf -- "${job_tmp}" ;; esac
    exit "${code}"
}
trap cleanup EXIT
git rev-parse HEAD > "${logs}/commit.txt"
baseline=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/enh7_baseline
bash run_whamr8k_bimamba_ctf_concat_s.sh --stage 5 --stop_stage 5 --ngpu 0 \
    --baseline_recipe "${baseline}" --python "${CONDA_PREFIX}/bin/python"
python local/check_whamr8k_baseline_match.py \
    --config conf/tuning/train_enh_tflocoformer_s_whamr8k_bimamba_ctf_concat.yaml \
    --baseline-config "${baseline}/exp/enh_train_tflocoformer_s_whamr_8k_1gpu_batch4/config.yaml" \
    --baseline-stats "${baseline}/../enh1/exp/enh_stats_8k" \
    --inputs exp/bimamba_whamr8k_baseline_inputs \
    --output exp/bimamba_whamr8k_baseline_match.json
