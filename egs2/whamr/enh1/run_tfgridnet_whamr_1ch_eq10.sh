#!/usr/bin/env bash
# Run in a dedicated tmux session on tensor; TMPDIR must be a private /tmp/nitsu directory.
set -euo pipefail
cd "$(dirname "$0")"
recipe=$PWD
repo_root=$(cd ../../.. && pwd)
source_recipe=/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/enh1
run_dir=${TFGRIDNET_RUN_DIR:-${source_recipe}/exp_tfgridnet_whamr_1ch_eq10_20261101}
python_bin=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
config=${recipe}/conf/tuning/train_tfgridnet_whamr_1ch_eq10.yaml
[[ $(hostname -s) == tensor ]] || { echo "This launcher targets tensor GPU 3"; exit 1; }
[[ ${TMPDIR:-} == /tmp/nitsu/whamr-eq10.* && -d ${TMPDIR} && -O ${TMPDIR} ]] || exit 1
mkdir -p "${run_dir}" "${TMPDIR}/numba" "${TMPDIR}/cache"
exec >> "${run_dir}/launcher.log" 2>&1
phase=starting
finish() {
    code=$?
    printf '%s phase=%s exit=%s\n' "$(date -Is)" "$phase" "$code" > "${run_dir}/run.status"
    rm -rf -- "${TMPDIR}"
}
trap finish EXIT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "$python_bin"):${PATH}"
export NUMBA_CACHE_DIR="${TMPDIR}/numba" XDG_CACHE_HOME="${TMPDIR}/cache"
export MPLCONFIGDIR="${TMPDIR}/cache/matplotlib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=3
check_gpu() {
    "${python_bin}" - <<'PY'
import subprocess
uuid = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row.split(',')[0].strip() == uuid for row in apps.splitlines()):
    raise SystemExit('tensor GPU 3 is occupied; refusing to start')
memory = int(subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
if memory > 1000:
    raise SystemExit('Unexplained GPU memory use; refusing to start')
PY
}
git rev-parse HEAD > "${run_dir}/code_commit.txt"
hostname > "${run_dir}/host.txt"
printf '3\n' > "${run_dir}/gpu_index.txt"
if [[ ! -f ${run_dir}/data_audit.json ]]; then
    phase=preparing_data
    "${python_bin}" local/tfgridnet_whamr_eq10.py prepare --source-recipe "${source_recipe}" --output "${run_dir}"
fi
check_gpu
phase=gpu_check
printf '%s phase=%s\n' "$(date -Is)" "$phase" > "${run_dir}/run.status"
"${python_bin}" local/tfgridnet_whamr_eq10.py check --gpu --config "${config}" --output "${run_dir}" > "${run_dir}/smoke_gpu.log" 2>&1
check_gpu
common=(
    --ngpu 1 --num_nodes 1 --nj 4 --python "${python_bin}"
    --train_set tr_mix_both_reverb_min_8k --valid_set cv_mix_both_reverb_min_8k
    --test_sets tt_mix_both_reverb_min_8k --fs 8k --ref_num 2 --ref_channel 0
    --audio_format wav --dumpdir "${run_dir}/dump" --expdir "${run_dir}"
    --enh_stats_dir "${run_dir}/stats" --enh_exp "${run_dir}/seed0"
    --enh_config "${config}" --use_noise_ref false --use_dereverb_ref false
    --inference_model valid.loss.best.pth
)
phase=training
printf '%s phase=%s\n' "$(date -Is)" "$phase" > "${run_dir}/run.status"
bash enh.sh --stage 6 --stop_stage 6 "${common[@]}" > "${run_dir}/train_launcher.log" 2>&1
if grep -q 'gradients at all steps are invalid' "${run_dir}/seed0/train.log"; then
    phase=invalid_gradients
    exit 1
fi
test -f "${run_dir}/seed0/valid.loss.best.pth"
phase=inference
printf '%s phase=%s\n' "$(date -Is)" "$phase" > "${run_dir}/run.status"
check_gpu
bash enh.sh --stage 7 --stop_stage 7 "${common[@]}" --gpu_inference true --inference_nj 1 \
    --inference_args '--normalize_output_wav true' > "${run_dir}/inference.log" 2>&1
phase=scoring
printf '%s phase=%s\n' "$(date -Is)" "$phase" > "${run_dir}/run.status"
scoring_protocol='SI_SNR SDR STOI ESTOI'
scoring_opts='--use_pesq false'
if "${python_bin}" -c 'import pesq' >/dev/null 2>&1; then
    scoring_protocol+=' PESQ_NB'
    scoring_opts='--use_pesq true'
else
    printf 'PESQ package unavailable; PESQ remains unmeasured.\n' > "${run_dir}/pesq_unavailable.txt"
fi
bash enh.sh --stage 8 --stop_stage 8 "${common[@]}" --inference_nj 8 \
    --scoring_protocol "${scoring_protocol}" --scoring_opts "${scoring_opts}" > "${run_dir}/scoring.log" 2>&1
phase=complete
