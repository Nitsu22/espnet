#!/usr/bin/env bash
# Run from the isolated evaluation checkout's rir_2spk directory.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -p -5
#$ -N bi_rir_infer
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
module load cuda/11.8.0
set -euo pipefail
variant=${1:?Specify 2blocks or bilstm}
result_root=${2:?Specify an absolute output root}
case "${variant}" in 2blocks|bilstm) ;; *) exit 2 ;; esac
[[ ${result_root} == /* ]]
recipe=${SGE_O_WORKDIR:?}
cd "${recipe}"
repo=$(git rev-parse --show-toplevel)
source_recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${repo}:${source_recipe}/../damsep_clean/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bi_eval_numba_${JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/bi_eval_triton_${JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bi_eval_mpl_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"
output="${result_root}/${variant}"
mkdir -p "${output}"
trap 'code=$?; echo "$code" > "$output/inference_exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
experiment="${source_recipe}/exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}"
training_job=8935090
[[ ${variant} != bilstm ]] || training_job=8935092
[[ $(cat "${source_recipe}/exp/pooled_bimamba_ablation_train_jobs/job_${training_job}/exit_status") == 0 ]]
[[ $(readlink "${experiment}/valid.loss.best.pth") == 45epoch.pth ]]
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'
whamr="${source_recipe}/dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k"
# Real CUDA smoke check on the allocated MIG device before the full datasets.
python local/evaluate_two_speaker.py --stage inference --limit 2 \
    --experiment "${experiment}" --data "${whamr}" --output "${output}/smoke_whamr"
for condition in whamr but_clean but_noisy; do
    case "${condition}" in
        whamr) data=${whamr} ;;
        but_clean) data="${source_recipe}/../rir_1ch/dump_but_2spk/raw/tt_but_2spk_clean_reverb_min_16k" ;;
        but_noisy) data="${source_recipe}/../rir_1ch/dump_but_2spk/raw/tt_but_2spk_noisy_reverb_min_16k" ;;
    esac
    [[ $(wc -l < "${data}/wav.scp") == 3000 ]]
    python local/evaluate_two_speaker.py --stage inference \
        --experiment "${experiment}" --data "${data}" --output "${output}/${condition}"
done
python - "${output}" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
reports = {name: json.loads((root / name / 'inference_complete.json').read_text())
           for name in ('whamr', 'but_clean', 'but_noisy')}
assert all(report['count'] == 3000 for report in reports.values())
(root / 'all_inference_complete.json').write_text(json.dumps(reports, indent=2) + '\n')
PY
