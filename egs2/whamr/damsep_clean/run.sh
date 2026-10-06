#!/usr/bin/env bash
set -euo pipefail

# Existing formatted dumps are linked into this recipe. Stages match enh.sh.
stage=5
stop_stage=8
ngpu=1
nj=4
python=python3
config=conf/tuning/train_damsep_nf_8k.yaml
dumpdir=dump_clean
reverb_dump=dump_reverb
statsdir=exp/damsep_stats_8k
expdir=exp/damsep_nf_8k
inference_model=valid.loss.best.pth
resume=true

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh
export PYTHONPATH="${MAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${statsdir}/.cache/numba}"
if (( stage < 5 || stop_stage > 8 || stage > stop_stage || ngpu < 0 || nj < 1 )); then
    echo "Use Stages 5-8, ngpu >= 0 and nj >= 1" >&2
    exit 2
fi
if (( ngpu < 1 && stage <= 7 && stop_stage >= 6 )); then
    echo "Stages 6 and 7 require at least one allocated GPU" >&2
    exit 2
fi

train_dir=${dumpdir}/raw/tr_mix_clean_reverb_min_8k
valid_dir=${dumpdir}/raw/cv_mix_clean_reverb_min_8k
test_dir=${dumpdir}/raw/tt_mix_clean_reverb_min_8k
args=()
for field in speech_mix speech_ref1 speech_ref2 speech_reverb1 speech_reverb2; do
    case ${field} in
        speech_mix) train_scp=${train_dir}/wav.scp; valid_scp=${valid_dir}/wav.scp ;;
        speech_ref1) train_scp=${train_dir}/spk1.scp; valid_scp=${valid_dir}/spk1.scp ;;
        speech_ref2) train_scp=${train_dir}/spk2.scp; valid_scp=${valid_dir}/spk2.scp ;;
        speech_reverb1|speech_reverb2)
            speaker=${field#speech_reverb}
            train_scp=${reverb_dump}/raw/tr_mix_both_reverb_min_8k/spk${speaker}_reverb.scp
            valid_scp=${reverb_dump}/raw/cv_mix_both_reverb_min_8k/spk${speaker}_reverb.scp ;;
    esac
    args+=(--train_data_path_and_name_and_type "${train_scp},${field},sound")
    args+=(--valid_data_path_and_name_and_type "${valid_scp},${field},sound")
done

if (( stage <= 5 && stop_stage >= 5 )); then
    mkdir -p "${statsdir}/logdir"
    "${python}" local/check_linked_dump.py --baseline-dump "${dumpdir}" \
        --reverb-dump "${reverb_dump}" --workers "${nj}" \
        --report "${statsdir}/data_check.json"
    train_keys=(); valid_keys=()
    for ((j=1; j<=nj; j++)); do
        train_keys+=("${statsdir}/logdir/train.${j}.scp")
        valid_keys+=("${statsdir}/logdir/valid.${j}.scp")
    done
    utils/split_scp.pl "${train_dir}/wav.scp" "${train_keys[@]}"
    utils/split_scp.pl "${valid_dir}/wav.scp" "${valid_keys[@]}"
    # Standard ESPnet shape collection skips network construction and feature
    # extraction via model_conf.extract_feats_in_collect_stats=false.
    ${train_cmd} JOB=1:"${nj}" "${statsdir}/logdir/stats.JOB.log" \
        "${python}" -m espnet2.bin.damsep_train --collect_stats true \
        --config "${config}" --ngpu 0 --num_workers 0 \
        --batch_size 1 --valid_batch_size 1 "${args[@]}" \
        --train_shape_file "${statsdir}/logdir/train.JOB.scp" \
        --valid_shape_file "${statsdir}/logdir/valid.JOB.scp" \
        --output_dir "${statsdir}/logdir/stats.JOB"
    stat_args=()
    for ((j=1; j<=nj; j++)); do stat_args+=(--input_dir "${statsdir}/logdir/stats.${j}"); done
    "${python}" -m espnet2.bin.aggregate_stats_dirs "${stat_args[@]}" \
        --skip_sum_stats --output_dir "${statsdir}"
fi

if (( stage <= 6 && stop_stage >= 6 )); then
    "${python}" -c 'import torch, mamba_ssm; assert torch.cuda.is_available(), "CUDA is unavailable"'
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd}" --log "${expdir}/train.log" --ngpu "${ngpu}" \
        --num_nodes 1 --init_file_prefix "${expdir}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.damsep_train --config "${config}" \
        --output_dir "${expdir}" --resume "${resume}" \
        --batch_size "${ngpu}" --valid_batch_size "${ngpu}" \
        --train_shape_file "${statsdir}/train/speech_mix_shape" \
        --valid_shape_file "${statsdir}/valid/speech_mix_shape" "${args[@]}"
fi

if (( stage <= 7 && stop_stage >= 7 )); then
    "${python}" -m espnet2.bin.damsep_inference \
        --train_config "${expdir}/config.yaml" --model_file "${expdir}/${inference_model}" \
        --wav_scp "${test_dir}/wav.scp" --output_dir "${expdir}/enhanced_tt" \
        --device cuda --normalize_output_wav true
fi

if (( stage <= 8 && stop_stage >= 8 )); then
    for branch in clean reverb; do
        refs=()
        for speaker in 1 2; do
            if [[ ${branch} == clean ]]; then
                ref=${test_dir}/spk${speaker}.scp
            else
                ref=${reverb_dump}/raw/tt_mix_both_reverb_min_8k/spk${speaker}_reverb.scp
            fi
            refs+=(--ref_scp "${ref}")
        done
        logdir=${expdir}/enhanced_tt/${branch}/scoring
        mkdir -p "${logdir}"
        keys=()
        for ((j=1; j<=nj; j++)); do keys+=("${logdir}/keys.${j}.scp"); done
        utils/split_scp.pl "${test_dir}/wav.scp" "${keys[@]}"
        ${train_cmd} JOB=1:"${nj}" "${logdir}/score.JOB.log" \
            "${python}" -m espnet2.bin.enh_scoring --output_dir "${logdir}/output.JOB" \
            --key_file "${logdir}/keys.JOB.scp" "${refs[@]}" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk1.scp" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk2.scp" --ref_channel 0
        "${python}" local/summarize_scores.py --scoring-dir "${logdir}" --jobs "${nj}"
    done
fi
