#!/usr/bin/env bash
# Hold this job on evaluate_pair_gpu.sh's job ID.
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=04:00:00
#$ -N damsep_pair_score
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5

set -e
. /gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh
conda activate tf-locoformer
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nitsu_damsep_score_mpl_${JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
if (( $# == 0 )); then
    set -- /gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/damsep_clean/exp/damsep_nf_8k \
        "${PWD}/exp/damsep_nf_8k_clean_only"
fi
printf 'Scoring code commit: %s\n' "$(git rev-parse HEAD)"
for expdir in "$@"; do
    # A hold only guarantees termination: reject partial/failed inference.
    python3 - "${expdir}" <<'PY'
import json
import sys
from pathlib import Path

expdir = Path(sys.argv[1])
expected = [line.split()[0] for line in Path(
    "dump_clean/raw/tt_mix_clean_reverb_min_8k/wav.scp"
).read_text().splitlines()]
report = json.loads((expdir / "enhanced_tt/inference.json").read_text())
if report["count"] != len(expected) or report["sample_rate"] != 8000:
    raise ValueError("Inference report does not cover the test set")
if not report["normalize_output_wav"] or len(expected) != len(set(expected)):
    raise ValueError("Unexpected inference normalization or duplicate test IDs")
branches = ("clean",) if report["clean_only"] else ("clean", "reverb")
for branch in branches:
    for speaker in (1, 2):
        scp = expdir / "enhanced_tt" / branch / f"spk{speaker}.scp"
        rows = [line.split(maxsplit=1) for line in scp.read_text().splitlines()]
        if [row[0] for row in rows] != expected:
            raise ValueError(f"Incomplete inference SCP: {scp}")
        if any(not Path(row[1]).is_file() for row in rows):
            raise ValueError(f"Missing inferred waveform: {scp}")
print(f"Validated {len(expected)} test utterances: {expdir}", flush=True)
PY
    bash run.sh --stage 8 --stop_stage 8 --ngpu 0 --nj 4 --expdir "${expdir}"
done
