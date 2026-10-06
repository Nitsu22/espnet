#!/usr/bin/env bash
set -euo pipefail

# Run from this recipe directory with the tf-locoformer environment activated.
stage=1
stop_stage=4
ngpu=1
nj=4
python=python3
config=conf/tuning/train_damsep_nf_8k.yaml
dumpdir=dump_nf_8k_min
expdir=exp/damsep_nf_8k
source_recipe=../enh4_clean
whamr_root=../enh1/data/whamr/2speakers/wav8k/min
reverb_dump=
inference_model=valid.loss.best.pth
resume=true

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh
export PYTHONPATH="${MAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${expdir}/.cache/numba}"
if (( ngpu < 1 || nj < 1 )); then
    echo "DAMSEP training/inference requires CUDA; ngpu and nj must be positive" >&2
    exit 2
fi
train_dir=${dumpdir}/raw/tr_mix_clean_reverb_min_8k
valid_dir=${dumpdir}/raw/cv_mix_clean_reverb_min_8k
test_dir=${dumpdir}/raw/tt_mix_clean_reverb_min_8k

if (( stage <= 1 && stop_stage >= 1 )); then
    prep_args=()
    if [[ -n ${reverb_dump} ]]; then prep_args+=(--reverb-dump "${reverb_dump}"); fi
    "${python}" local/prepare_nf_dump.py --source "${source_recipe}" \
        --whamr-root "${whamr_root}" --output "${dumpdir}" --verify-existing "${prep_args[@]}"
fi

if (( stage <= 2 && stop_stage >= 2 )); then
    "${python}" -c 'import torch, mamba_ssm; assert torch.cuda.is_available(), "CUDA is unavailable"'
    args=()
    for field in speech_mix speech_ref1 speech_ref2 speech_reverb1 speech_reverb2; do
        case ${field} in
            speech_mix) scp=wav ;;
            speech_ref1) scp=spk1 ;;
            speech_ref2) scp=spk2 ;;
            speech_reverb1) scp=spk1_reverb ;;
            speech_reverb2) scp=spk2_reverb ;;
        esac
        args+=(--train_data_path_and_name_and_type "${train_dir}/${scp}.scp,${field},sound")
        args+=(--valid_data_path_and_name_and_type "${valid_dir}/${scp}.scp,${field},sound")
    done
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd}" --log "${expdir}/train.log" --ngpu "${ngpu}" \
        --num_nodes 1 --init_file_prefix "${expdir}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.damsep_train --config "${config}" \
        --output_dir "${expdir}" --resume "${resume}" \
        --batch_size "${ngpu}" --valid_batch_size "${ngpu}" \
        --train_shape_file "${train_dir}/speech_mix_shape" \
        --valid_shape_file "${valid_dir}/speech_mix_shape" "${args[@]}"
fi

if (( stage <= 3 && stop_stage >= 3 )); then
    "${python}" -m espnet2.bin.damsep_inference \
        --train_config "${expdir}/config.yaml" --model_file "${expdir}/${inference_model}" \
        --wav_scp "${test_dir}/wav.scp" --output_dir "${expdir}/enhanced_tt" \
        --device cuda --normalize_output_wav true
fi

if (( stage <= 4 && stop_stage >= 4 )); then
    for branch in clean reverb; do
        refs=()
        for speaker in 1 2; do
            if [[ ${branch} == clean ]]; then ref=spk${speaker}; else ref=spk${speaker}_reverb; fi
            refs+=(--ref_scp "${test_dir}/${ref}.scp")
        done
        logdir=${expdir}/enhanced_tt/${branch}/scoring
        mkdir -p "${logdir}"
        keys=()
        for ((j=1; j<=nj; j++)); do keys+=("${logdir}/keys.${j}.scp"); done
        utils/split_scp.pl "${test_dir}/wav.scp" "${keys[@]}"
        # The score jobs use the same ESPnet scorer as enh4_clean, on CPUs.
        ${train_cmd} JOB=1:"${nj}" "${logdir}/score.JOB.log" \
            "${python}" -m espnet2.bin.enh_scoring --output_dir "${logdir}/output.JOB" \
            --key_file "${logdir}/keys.JOB.scp" "${refs[@]}" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk1.scp" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk2.scp" --ref_channel 0
        "${python}" local/summarize_scores.py --scoring-dir "${logdir}" --jobs "${nj}"
    done
fi
