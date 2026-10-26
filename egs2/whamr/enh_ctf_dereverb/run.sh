#!/usr/bin/env bash
# Execute inside an existing environment/allocation; this never submits jobs.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"
main_root=$(cd ../../.. && pwd)

phase=stage1
stage=6
stop_stage=6
paired_dump=../enh_rir/dump_ctf_joint
baseline_root=../enh1
source_data_root=../enh1/data
baseline_dump=dump/raw
baseline_stats=exp/enh_stats_8k
enh_exp=
enh_config=
ngpu=1
python=python3
cuda_cmd=run.pl
resume=false
init_param=
allow_scratch=false
dry_run=false
help_message="Usage: bash run.sh [options]
  --phase stage1|stage2|stage2_nocg|b1
  --stage 1 --stop_stage 1    Verify source audio and prepare a NEW paired dump
  --stage 6 --stop_stage 6    Train (default; existing paired dump required)
  --paired_dump PATH         Verified Direct/Reverb dump with manifest and stats
  --enh_exp PATH             Separate output directory (required for batch changes)
  --init_param PATH          Stage 1 model weights for Stage 2, e.g. valid.loss.best.pth
  --resume true              Continue this output's checkpoint; do not reinitialize
  --allow_scratch true       Explicit Stage 2 ablation from random initialization
  --ngpu 1 --python python3  Use caller's existing environment/allocation
  --dry_run true             Validate inputs and print the training command"
. "${main_root}/egs2/TEMPLATE/asr1/utils/parse_options.sh"

case ${phase} in
    stage1|stage2|stage2_nocg|b1) ;;
    *) echo "Unknown phase: ${phase}" >&2; exit 2 ;;
esac
for flag in resume allow_scratch dry_run; do
    if [[ ${!flag} != true && ${!flag} != false ]]; then
        echo "--${flag} must be true or false" >&2; exit 2
    fi
done
if [[ ! ${stage} =~ ^[0-9]+$ || ! ${stop_stage} =~ ^[0-9]+$ ||
      ! ${ngpu} =~ ^[0-9]+$ ]] ||
    (( stage > stop_stage )) ||
    [[ ${stage} != 1 && ${stage} != 6 ]] ||
    [[ ${stop_stage} != 1 && ${stop_stage} != 6 ]]; then
    echo "Use preparation stage 1 and/or training stage 6; ngpu must be nonnegative." >&2
    exit 2
fi
enh_config=${enh_config:-conf/train_ctf_${phase}.yaml}
enh_exp=${enh_exp:-exp/enh_train_ctf_${phase}_whamr_8k_${ngpu}gpu_batch4}
[[ -f ${enh_config} ]] || { echo "Missing config: ${enh_config}" >&2; exit 2; }

export PATH="${main_root}/egs2/TEMPLATE/asr1/utils:${PATH}"
export PYTHONPATH="${main_root}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONIOENCODING=UTF-8

if (( stage <= 1 && stop_stage >= 1 )); then
    if [[ ${dry_run} == true ]]; then
        echo "Preparation requires reading all paired audio; use --stage 6 for a command preview." >&2
        exit 2
    fi
    "${python}" ../enh_rir/local/prepare_ctf_joint_dump.py \
        --baseline-root "${baseline_root}" --source-data-root "${source_data_root}" \
        --baseline-dump "${baseline_dump}" --baseline-stats "${baseline_stats}" \
        --output "${paired_dump}"
fi
if (( stage > 6 || stop_stage < 6 )); then exit 0; fi

"${python}" local/check_paired_dump.py "${paired_dump}"
if [[ ${resume} == true ]]; then
    [[ -f ${enh_exp}/checkpoint.pth ]] || {
        echo "No checkpoint to resume: ${enh_exp}/checkpoint.pth" >&2; exit 2;
    }
    [[ -z ${init_param} ]] || {
        echo "Use either --resume true or --init_param, not both." >&2; exit 2;
    }
elif [[ -e ${enh_exp}/checkpoint.pth ]]; then
    echo "Existing checkpoint: use --resume true or a new --enh_exp." >&2; exit 2
fi
if [[ -n ${init_param} && ! -f ${init_param%%:*} ]]; then
    echo "Missing initial model weights: ${init_param%%:*}" >&2; exit 2
fi
if [[ ${phase} == stage2* && ${resume} == false && -z ${init_param} &&
      ${allow_scratch} != true ]]; then
    echo "Stage 2 requires --init_param STAGE1_WEIGHTS (or explicit --allow_scratch true)." >&2
    exit 2
fi

data_args=()
for split in train valid; do
    dataset=tr_mix_both_reverb_min_8k
    if [[ ${split} == valid ]]; then dataset=cv_mix_both_reverb_min_8k; fi
    for entry in 'wav.scp speech_mix' 'spk1.scp speech_ref1' 'spk2.scp speech_ref2' \
                 'spk1_reverb.scp speech_reverb1' 'spk2_reverb.scp speech_reverb2'; do
        read -r scp name <<< "${entry}"
        data_args+=("--${split}_data_path_and_name_and_type"
                    "${paired_dump}/raw/${dataset}/${scp},${name},sound")
    done
    for name in speech_mix speech_ref1 speech_ref2; do
        data_args+=("--${split}_shape_file" "${paired_dump}/stats/${split}/${name}_shape")
    done
done
train_cmd=("${python}" -m espnet2.bin.enh_train --config "${enh_config}"
           --output_dir "${enh_exp}" --ngpu "${ngpu}" --resume "${resume}"
           --fold_length 200000 --fold_length 200000 --fold_length 200000)
if [[ -n ${init_param} ]]; then train_cmd+=(--init_param "${init_param}"); fi
train_cmd+=("${data_args[@]}")
command=("${python}" -m espnet2.bin.launch --cmd "${cuda_cmd}"
         --log "${enh_exp}/train.log" --ngpu "${ngpu}" --num_nodes 1
         --init_file_prefix "${enh_exp}/.dist_init_" --multiprocessing_distributed true --
         "${train_cmd[@]}")
if [[ ${dry_run} == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi

# Keep temporary/compiler/cache writes inside the user's permitted locations.
if [[ -e /tmp/nitsu && ! -O /tmp/nitsu ]]; then
    echo "/tmp/nitsu is not owned by this user; select a permitted temporary parent." >&2
    exit 2
fi
mkdir -p /tmp/nitsu
task_tmp=$(mktemp -d /tmp/nitsu/ctf-dereverb.XXXXXXXX)
trap 'rm -rf -- "${task_tmp}"' EXIT
export TMPDIR=${task_tmp}
export XDG_CACHE_HOME="${script_dir}/.cache"
export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
"${command[@]}"
