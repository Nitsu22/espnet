#!/usr/bin/env bash
# Submit with -hold_jid pointing to this variant's actual inference job.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:15:00
#$ -p -5
#$ -N bi_rir_score
#$ -o qsub_logs
#$ -e qsub_logs
set -e
source /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
variant=${1:?Specify 2blocks, bilstm or drr}
result_root=${2:?Specify an absolute output root}
case "${variant}" in 2blocks|bilstm|drr) ;; *) exit 2 ;; esac
[[ ${result_root} == /* ]]
recipe=${SGE_O_WORKDIR:?}
cd "${recipe}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/bi_score_numba_${JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"
source_recipe=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/rir_2spk
output="${result_root}/${variant}"
mkdir -p "${output}"
trap 'code=$?; echo "$code" > "$output/scoring_exit_status"; date -Is' EXIT
date -Is
hostname
git rev-parse HEAD
# A dependency only waits for termination; require successful full inference.
[[ $(cat "${output}/inference_exit_status") == 0 ]]
test -s "${output}/all_inference_complete.json"
training_recipe=${source_recipe}
checkpoint=valid.loss.best.pth
if [[ ${variant} == drr ]]; then
    training_recipe=${recipe}
    checkpoint=valid.loss_sweep.best.pth
fi
experiment="${training_recipe}/exp/rir_train_pooled_bimamba_2spk_nf_16k_sweep_v2_${variant}"
for condition in whamr but_clean but_noisy; do
    extra=()
    case "${condition}" in
        whamr) data="${source_recipe}/dump_nf_2spk_16k_min/raw/tt_rir_2spk_nf_min_16k" ;;
        but_clean) data="${source_recipe}/../rir_1ch/dump_but_2spk/raw/tt_but_2spk_clean_reverb_min_16k"; extra=(--also_polarity_aligned) ;;
        but_noisy) data="${source_recipe}/../rir_1ch/dump_but_2spk/raw/tt_but_2spk_noisy_reverb_min_16k"; extra=(--also_polarity_aligned) ;;
    esac
    python local/evaluate_two_speaker.py --stage score --experiment "${experiment}" --checkpoint "${checkpoint}" \
        --data "${data}" --output "${output}/${condition}" "${extra[@]}"
done
python - "${output}" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys
root = Path(sys.argv[1])
reports = {}
for condition in ('whamr', 'but_clean', 'but_noisy'):
    names = ('score',) if condition == 'whamr' else ('score', 'score_polarity_aligned')
    reports[condition] = {}
    for name in names:
        folder = root / condition / name
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['num_utts'] == 3000 and summary['num_source_pairs'] == 6000
        # Keep metric implementations unchanged, and expose NaN exclusions.
        with (folder / 'per_source.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        valid_counts = {key: sum(math.isfinite(float(row[key])) for row in rows)
                        for key in ('rmse', 'corr', 'rt60_err', 'drr_err', 'c50_err')}
        (folder / 'valid_metric_counts.json').write_text(json.dumps(valid_counts, indent=2) + '\n')
        reports[condition][name] = dict(summary=summary, valid_counts=valid_counts)
(root / 'evaluation_complete.json').write_text(json.dumps(reports, indent=2) + '\n')
PY
