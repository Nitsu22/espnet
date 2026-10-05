#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

stage=1
stop_stage=6
ngpu=4
baseline_root=../enh1
source_data_root=../enh1/data
baseline_dump=dump/raw
baseline_stats=exp/enh_stats_8k
dumpdir=dump_ctf_joint
enh_exp=exp/enh_train_enh_tflocoformer_small_nocashe_ctf_joint_4gpu
enh_config=conf/tuning/train_enh_tflocoformer_small_nocashe_ctf_joint_4gpu.yaml
inference_model=valid.loss_clean_pit.best.pth
resume=false
python=python

. ./utils/parse_options.sh
. ./path.sh
. ./cmd.sh
export PYTHONPATH="${MAIN_ROOT}:${PYTHONPATH:-}"

if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
    "${python}" local/prepare_ctf_joint_dump.py \
        --baseline-root "${baseline_root}" --source-data-root "${source_data_root}" \
        --baseline-dump "${baseline_dump}" --baseline-stats "${baseline_stats}" \
        --output "${dumpdir}"
fi

if [[ ${stage} -le 6 && ${stop_stage} -ge 6 ]]; then
    if [[ ! -f ${dumpdir}/manifest.json ]]; then
        echo "Prepare and verify the paired baseline dump with --stage 1 --stop_stage 1." >&2
        exit 1
    fi
    if [[ ${resume} == false && -e ${enh_exp}/checkpoint.pth ]]; then
        echo "Existing checkpoint: use --resume true or a new --enh_exp." >&2
        exit 1
    fi
    args=()
    for split in train valid; do
        if [[ ${split} == train ]]; then
            dataset=tr_mix_both_reverb_min_8k
        else
            dataset=cv_mix_both_reverb_min_8k
        fi
        for entry in 'wav.scp speech_mix' 'spk1.scp speech_ref1' 'spk2.scp speech_ref2' \
                     'spk1_reverb.scp speech_reverb1' 'spk2_reverb.scp speech_reverb2'; do
            read -r scp name <<< "${entry}"
            args+=("--${split}_data_path_and_name_and_type" "${dumpdir}/raw/${dataset}/${scp},${name},sound")
        done
        for name in speech_mix speech_ref1 speech_ref2; do
            args+=("--${split}_shape_file" "${dumpdir}/stats/${split}/${name}_shape")
        done
    done
    # Same 80,000-sample folding thresholds and global batch as the baseline.
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd}" --log "${enh_exp}/train.log" --ngpu "${ngpu}" --num_nodes 1 \
        --init_file_prefix "${enh_exp}/.dist_init_" --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.enh_train --config "${enh_config}" \
        --output_dir "${enh_exp}" --ngpu "${ngpu}" --resume "${resume}" \
        --fold_length 80000 --fold_length 80000 --fold_length 80000 \
        "${args[@]}"
fi

if [[ ${stage} -le 8 && ${stop_stage} -ge 7 ]]; then
    eval_stage=7
    if [[ ${stage} -gt 7 ]]; then eval_stage=${stage}; fi
    ./enh.sh --stage "${eval_stage}" --stop_stage "${stop_stage}" \
        --train_set tr_mix_both_reverb_min_8k --valid_set cv_mix_both_reverb_min_8k \
        --test_sets tt_mix_both_reverb_min_8k --fs 8k --ref_num 2 \
        --enh_config "${enh_config}" --enh_exp "${enh_exp}" --dumpdir "${dumpdir}" \
        --use_dereverb_ref false --use_noise_ref false --audio_format wav \
        --inference_model "${inference_model}" --gpu_inference true
fi
