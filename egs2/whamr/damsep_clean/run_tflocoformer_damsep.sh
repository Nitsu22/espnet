#!/usr/bin/env bash
set -euo pipefail

stage=6
stop_stage=6
ngpu=1
nj=4
variant=clean_reverb
expdir=
batch_size=4
accum_grad=1
resume=true
inference_model=valid.loss_clean.best.pth
statsdir=exp/tflocoformer_damsep_stats_8k
dumpdir=dump_whamr
reverb_dump=dump_reverb
. utils/parse_options.sh
case ${variant} in clean_reverb|three_loss) ;; *) echo "Unknown variant" >&2; exit 2 ;; esac
if (( stage < 6 || stop_stage > 8 || stage > stop_stage || ngpu < 0 || nj < 1 || batch_size < 1 || accum_grad < 1 )); then
    echo "Use Stages 6-8 and valid resource/batch options" >&2; exit 2
fi
if (( stage <= 7 && ngpu != 1 )); then
    echo "Stages 6 and 7 require exactly one allocated GPU" >&2; exit 2
fi
. ./path.sh
. ./cmd.sh
export PATH="${PATH}:/usr/sbin:/sbin"
export PYTHONPATH="${MAIN_ROOT}:${PWD}/.deps/mamba-cu118-torch21-py310${PYTHONPATH:+:${PYTHONPATH}}"
config=conf/tuning/train_tflocoformer_damsep_whamr_8k_${variant}.yaml
expdir=${expdir:-exp/tflocoformer_damsep_whamr_8k_${variant}}
train_dir=${dumpdir}/raw/tr_mix_both_reverb_min_8k
valid_dir=${dumpdir}/raw/cv_mix_both_reverb_min_8k
test_dir=${dumpdir}/raw/tt_mix_both_reverb_min_8k
args=()
for field in speech_mix speech_ref1 speech_ref2 speech_reverb1 speech_reverb2; do
    case ${field} in
        speech_mix) tr=${train_dir}/wav.scp; cv=${valid_dir}/wav.scp ;;
        speech_ref1|speech_ref2)
            speaker=${field#speech_ref}
            tr=${train_dir}/spk${speaker}.scp; cv=${valid_dir}/spk${speaker}.scp ;;
        speech_reverb1|speech_reverb2)
            speaker=${field#speech_reverb}
            tr=${reverb_dump}/raw/tr_mix_both_reverb_min_8k/spk${speaker}_reverb.scp
            cv=${reverb_dump}/raw/cv_mix_both_reverb_min_8k/spk${speaker}_reverb.scp ;;
    esac
    args+=(--train_data_path_and_name_and_type "${tr},${field},sound")
    args+=(--valid_data_path_and_name_and_type "${cv},${field},sound")
done
if (( stage <= 6 && stop_stage >= 6 )); then
    python3 -m espnet2.bin.launch --cmd "${cuda_cmd}" \
        --log "${expdir}/train.log" --ngpu 1 --num_nodes 1 \
        --init_file_prefix "${expdir}/.dist_init_" --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.damsep_train --config "${config}" \
        --output_dir "${expdir}" --resume "${resume}" \
        --batch_size "${batch_size}" --valid_batch_size 1 --accum_grad "${accum_grad}" \
        --train_shape_file "${statsdir}/train/speech_mix_shape" \
        --valid_shape_file "${statsdir}/valid/speech_mix_shape" "${args[@]}"
fi
if (( stage <= 7 && stop_stage >= 7 )); then
    python3 -m espnet2.bin.damsep_inference --train_config "${expdir}/config.yaml" \
        --model_file "${expdir}/${inference_model}" --wav_scp "${test_dir}/wav.scp" \
        --output_dir "${expdir}/enhanced_tt" --device cuda --normalize_output_wav true
fi
if (( stage <= 8 && stop_stage >= 8 )); then
    # Both conditions have clean and reverberant audio outputs, regardless of CTF.
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
            python3 -m espnet2.bin.enh_scoring --output_dir "${logdir}/output.JOB" \
            --key_file "${logdir}/keys.JOB.scp" "${refs[@]}" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk1.scp" \
            --inf_scp "${expdir}/enhanced_tt/${branch}/spk2.scp" --ref_channel 0
        python3 local/summarize_scores.py --scoring-dir "${logdir}" --jobs "${nj}"
    done
fi
