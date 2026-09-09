#!/usr/bin/env bash
# Stage 1 only: generate stereo WHAMR audio, physical RIRs, and signal SCPs.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
stage=1
stop_stage=1
output_dir=${recipe}/data_rir_plus
min_free_gib=100
sample_rates="8k 16k"
data_lengths="min max"
splits="tr cv tt"
limit=
estimate_only=false
. "${recipe}/utils/parse_options.sh"
if [[ ${stage} != 1 || ${stop_stage} != 1 || $# != 0 ]]; then
    echo 'This launcher supports --stage 1 --stop_stage 1 only.' >&2
    exit 2
fi
extra=()
[[ -z ${limit} ]] || extra+=(--limit "${limit}")
case ${estimate_only} in
    true) extra+=(--estimate-only) ;;
    false) ;;
    *) echo '--estimate_only must be true or false' >&2; exit 2 ;;
esac
read -r -a rates <<< "${sample_rates}"
read -r -a lengths <<< "${data_lengths}"
read -r -a split_list <<< "${splits}"
bash "${recipe}/run_generate_rir_plus.sh" \
    --output-dir "${output_dir}" --min-free-gib "${min_free_gib}" \
    --sample-rates "${rates[@]}" --data-lengths "${lengths[@]}" \
    --splits "${split_list[@]}" "${extra[@]}"
