#!/usr/bin/env bash
# Test-only paired evaluation; never trains or processes CV.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
test_dir=
baseline_exp=
conditioned_exp=
output_dir=
stage=7
stop_stage=8
resume=false
python=python3
while (( $# )); do
    case "$1" in
        --test_dir|--baseline_exp|--conditioned_exp|--output_dir|--stage|--stop_stage|--resume|--python)
            name=${1#--}; [[ $# -ge 2 ]] || exit 2; printf -v "${name}" '%s' "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n ${test_dir} && -n ${baseline_exp} && -n ${conditioned_exp} && -n ${output_dir} ]]
[[ ${stage} =~ ^[78]$ && ${stop_stage} =~ ^[78]$ && ${stage} -le ${stop_stage} ]]
[[ ${resume} == true || ${resume} == false ]]
checker=$(dirname "${BASH_SOURCE[0]}")/check_same_speaker_evaluation.py
opts=()
[[ ${resume} == false ]] || opts+=(--resume)
"${python}" "${checker}" prepare --test-dir "${test_dir}" \
    --baseline-exp "${baseline_exp}" --conditioned-exp "${conditioned_exp}" \
    --output-dir "${output_dir}" "${opts[@]}"
export TMPDIR="${output_dir}/.tmp"
export NUMBA_CACHE_DIR="${TMPDIR}/numba" TRITON_CACHE_DIR="${TMPDIR}/triton"
export MPLCONFIGDIR="${TMPDIR}/matplotlib" XDG_CACHE_HOME="${TMPDIR}/xdg"
mkdir -p "${TMPDIR}"
cleanup() {
    status=$?
    if [[ ${TMPDIR} == "${output_dir}/.tmp" && -d ${TMPDIR} && ! -L ${TMPDIR} ]]; then
        rm -r -- "${TMPDIR}"
    fi
    exit "${status}"
}
trap cleanup EXIT
for model in baseline conditioned; do
    out=${output_dir}/${model}
    if (( stage <= 7 && stop_stage >= 7 )); then
        if [[ ${resume} == true ]] && "${python}" "${checker}" check-inference "${output_dir}" "${model}" > /dev/null 2>&1; then
            echo "Verified existing complete ${model} inference; skipping"
        else
            paths=$("${python}" "${checker}" paths "${output_dir}" "${model}")
            mapfile -t model_paths <<< "${paths}"
            mkdir -p "${out}"
            "${python}" -m espnet2.bin.enh_inference \
                --ngpu 1 --fs 8000 --batch_size 1 \
                --train_config "${model_paths[0]}" --model_file "${model_paths[1]}" \
                --data_path_and_name_and_type "${test_dir}/wav.scp,speech_mix,sound" \
                --key_file "${test_dir}/wav.scp" --output_dir "${out}" \
                > "${out}/inference.log" 2>&1
            "${python}" "${checker}" check-inference "${output_dir}" "${model}"
        fi
    fi
    if (( stage <= 8 && stop_stage >= 8 )); then
        "${python}" "${checker}" check-inference "${output_dir}" "${model}"
        if [[ ${resume} == true ]] && "${python}" "${checker}" check-score "${output_dir}" "${model}" > /dev/null 2>&1; then
            echo "Verified existing complete ${model} scores; skipping"
        else
            "${python}" -m espnet2.bin.enh_scoring \
                --ref_scp "${test_dir}/spk1.scp" --ref_scp "${test_dir}/spk2.scp" \
                --inf_scp "${out}/spk1.scp" --inf_scp "${out}/spk2.scp" \
                --key_file "${test_dir}/wav.scp" --ref_channel 0 \
                --output_dir "${out}/score" > "${out}/scoring.log" 2>&1
            "${python}" "${checker}" check-score "${output_dir}" "${model}"
        fi
    fi
done
if (( stop_stage >= 8 )); then
    "${python}" "${checker}" summary "${output_dir}"
fi
