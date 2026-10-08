#!/usr/bin/env bash
# Submit from rir_2spk. This is a charged measurement, not an uncharged trial.
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=00:15:00
#$ -p -5
#$ -N bimamba_abl_check
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
cd "${recipe}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${recipe}/../../..:${recipe}/../damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_profile_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/bimamba_profile_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bimamba_profile_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
logs="exp/pooled_bimamba_ablations_profile/job_${JOB_ID}"
mkdir -p "${logs}"
trap 'code=$?; echo "$code" > "$logs/exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
for variant in ffn_only freq_before_pool 2blocks bilstm; do
    python local/check_two_speaker_training.py \
        --config "conf/tuning/train_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}.yaml" \
        --data-dir dump_nf_2spk_16k_min/raw/tr_rir_2spk_nf_min_16k \
        --valid-data-dir dump_nf_2spk_16k_min/raw/cv_rir_2spk_nf_min_16k \
        --batch-size 4 --warmup-steps 3 --timed-steps 12 \
        --output "${logs}/${variant}.json" > "${logs}/${variant}.log" 2>&1
    cat "${logs}/${variant}.log"
done
