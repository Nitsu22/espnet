#!/usr/bin/env bash
# Stage 1: immutable reference dump. Stage 2: score existing model predictions.
set -eo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
cd "${recipe}"
stage=1
stop_stage=1
sample_rate=16000
rir_length=
filter_length=
sweep_duration=8.192
data=
reference_dir=
prediction=
model_label=checkpoint
out_dir=
python=python
workers=2
regularization=0.000001
. utils/parse_options.sh
. ./path.sh
set -u
case ${sample_rate} in
    8000) rate_tag=8k ;;
    16000) rate_tag=16k ;;
    *) echo "Expected --sample_rate 8000 or 16000" >&2; exit 2 ;;
esac
rir_length=${rir_length:-$((sample_rate * 2))}
filter_length=${filter_length:-$((sample_rate * 2))}
data=${data:-dump_nf_2spk_${rate_tag}_min/raw/tt_rir_2spk_nf_min_${rate_tag}}
reference_dir=${reference_dir:-dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_${rate_tag}}
if [[ -z ${out_dir} ]]; then
    out_dir=exp/ls_direct_reverb_evaluation
    [[ ${sample_rate} == 16000 ]] || out_dir+=_${rate_tag}
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${recipe}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
if (( stage <= 1 && stop_stage >= 1 )); then
    if [[ -e ${reference_dir} ]]; then
        "${python}" local/prepare_ls_rir_reference.py --mode validate --output "${reference_dir}"
        "${python}" - "${reference_dir}" "${regularization}" "${data}" \
            "${sample_rate}" "${rir_length}" "${filter_length}" "${sweep_duration}" <<'PY'
import json
import hashlib
from pathlib import Path
import sys
from espnet2.rir.ls_rir import LSRIRConfig
spec = json.loads((Path(sys.argv[1]) / 'preparation.json').read_text())
expected = LSRIRConfig(
    regularization=float(sys.argv[2]), sample_rate=int(sys.argv[4]),
    rir_length=int(sys.argv[5]), filter_length=int(sys.argv[6]),
    sweep_duration=float(sys.argv[7]),
).as_dict()
if (spec['config'] != expected or spec['channel'] != 0
        or Path(spec['source_data']).resolve() != Path(sys.argv[3]).resolve()):
    raise ValueError('Existing reference settings differ; use a new reference_dir')
for name, checksum in spec['source_index_sha256'].items():
    if hashlib.sha256((Path(sys.argv[3]) / name).read_bytes()).hexdigest() != checksum:
        raise ValueError('Source indexes changed; use a new reference_dir')
PY
    else
        "${python}" local/prepare_ls_rir_reference.py --data "${data}" \
            --output "${reference_dir}" --workers "${workers}" --regularization "${regularization}" \
            --sample_rate "${sample_rate}" --rir_length "${rir_length}" \
            --filter_length "${filter_length}" --sweep_duration "${sweep_duration}"
    fi
fi
if (( stage <= 2 && stop_stage >= 2 )); then
    [[ -n ${prediction} ]] || { echo "Stage 2 requires --prediction RIR_SCP_OR_DIR" >&2; exit 2; }
    "${python}" local/evaluate_ls_direct_reverb.py --reference_dir "${reference_dir}" \
        --prediction "${model_label}" "${prediction}" --out_dir "${out_dir}"
fi
