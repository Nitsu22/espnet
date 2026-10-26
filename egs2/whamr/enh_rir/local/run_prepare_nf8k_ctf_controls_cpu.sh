#!/usr/bin/env bash
# CPU-only preparation. Run in a private tmux session; never launches a model.
set -euo pipefail
recipe=${1:?absolute original enh_rir recipe}
jobdir=${2:?absolute private log directory}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
case ${recipe}:${jobdir} in
    /net/midgar/work2/nitsu/*:/net/midgar/work2/nitsu/*) ;;
    *) echo 'Use explicit paths in the authorized personal storage' >&2; exit 2 ;;
esac
test -d "${jobdir}"
test ! -e "${jobdir}/exit_status"
test ! -e "${recipe}/dump_nf8k_ctf_controls_seed0"
task_tmp=$(mktemp -d "${jobdir}/tmp.XXXXXX")
export TMPDIR=${task_tmp} XDG_CACHE_HOME=${task_tmp}/cache
export NUMBA_CACHE_DIR=${task_tmp}/numba MPLCONFIGDIR=${task_tmp}/matplotlib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
cleanup() {
    code=$?
    trap - EXIT
    printf '%s\n' "${code}" > "${jobdir}/exit_status"
    case ${task_tmp} in "${jobdir}"/tmp.*) rm -r -- "${task_tmp}" ;; esac
    date -Is
    exit "${code}"
}
trap cleanup EXIT
date -Is
hostname
git -C "${script_dir}" rev-parse HEAD | tee "${jobdir}/commit.txt"
python=/home/kslab/nitsu/.conda/envs/tf-locoformer/bin/python
args=(
    --baseline-test "${recipe}/exp/bimamba_nf8k_baseline_inputs/raw/tt_mix_clean_reverb_min_8k"
    --baseline-recipe "${recipe}/../enh7_baseline"
    --raw-root "${recipe}/data/wsj0/wsj0_wav"
    --physical-root "${recipe}/data_rir_plus"
    --csv-dir "${recipe}/whamr_scripts/data"
    --seed 0 --workers 4 --expected-count 3000
)
echo 'Pilot: generate and validate the first 10 scenes in both conditions'
"${python}" -B "${script_dir}/prepare_nf8k_ctf_controls.py" "${args[@]}" \
    --limit 10 --output "${jobdir}/pilot_seed0"
echo 'Full preparation: generate both conditioning-input dumps; no CTF/model inference'
"${python}" -B "${script_dir}/prepare_nf8k_ctf_controls.py" "${args[@]}" \
    --output "${recipe}/dump_nf8k_ctf_controls_seed0"
test -s "${recipe}/dump_nf8k_ctf_controls_seed0/preparation.json"
printf '%s\n' 'Both audio dumps generated; inference and scoring NOT started' > "${jobdir}/completed"
