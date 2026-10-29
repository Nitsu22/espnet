#!/usr/bin/env bash
# Run inside an allocation; the script never selects physical GPUs.
set -euo pipefail
stage=6
stop_stage=6
ngpu=1
python=python3
dumpdir=
condition=nf_whamr
variant=concat
enh_config=
enh_exp=
resume=false
inference_model=valid.loss.best.pth
baseline_recipe=../enh7_baseline
baseline_dump=../enh7_baseline/dump_clean
baseline_reverb_data=../enh4_clean/data
rir_recipe=../rir_2spk
rir_dump=../rir_2spk/dump_nf_2spk_8k_min
workers=4
. ./path.sh
. ./cmd.sh
. utils/parse_options.sh
case ${variant} in concat|film) ;; *) echo 'Unknown Oracle variant' >&2; exit 2 ;; esac
case ${condition} in
    nf_whamr) tag=nf8k; mixture=mix_clean_reverb_min_8k ;;
    whamr) tag=whamr8k; mixture=mix_both_reverb_min_8k
        [[ ${variant} == concat ]] || { echo 'WHAMR uses concat' >&2; exit 2; } ;;
    *) echo 'Unknown dataset condition' >&2; exit 2 ;;
esac
dumpdir=${dumpdir:-dump_oracle_ctf_${tag}}
enh_config=${enh_config:-conf/tuning/train_enh_tflocoformer_s_${tag}_oracle_ctf_${variant}.yaml}
enh_exp=${enh_exp:-exp/enh_train_tflocoformer_s_${tag}_oracle_ctf_${variant}_seed0}
export PYTHONPATH="${PWD}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
[[ $# == 0 ]] || { echo 'Unexpected positional arguments' >&2; exit 2; }
if (( stage < 1 || stage > stop_stage || stop_stage > 8 || ngpu < 0 )); then
    echo 'Use Stage 1 (precompute) or Stages 5–8' >&2; exit 2
fi
if (( stage <= 1 && stop_stage >= 1 )); then
    [[ ${condition} == nf_whamr ]] || { echo "Use qsub/prepare_whamr_oracle.sh for WHAMR" >&2; exit 2; }
    "${python}" local/prepare_oracle_ctf_dump.py --mode prepare \
        --baseline-dump "${baseline_dump}" --baseline-recipe "${baseline_recipe}" \
        --baseline-reverb-data "${baseline_reverb_data}" \
        --rir-dump "${rir_dump}" --rir-recipe "${rir_recipe}" \
        --workers "${workers}" --output "${dumpdir}"
fi
if (( stop_stage < 5 )); then exit 0; fi
"${python}" local/check_oracle_ctf_inputs.py --dump "${dumpdir}" --condition "${condition}"
if (( stage <= 6 && stop_stage >= 6 )); then
    [[ ${ngpu} == 1 ]] || { echo 'Training requires one allocated GPU' >&2; exit 2; }
    case ${resume} in
        true) test -s "${enh_exp}/checkpoint.pth" ;;
        false) [[ ! -e ${enh_exp}/checkpoint.pth ]] || { echo 'Use --resume true or a new --enh_exp' >&2; exit 2; } ;;
        *) echo 'resume must be true or false' >&2; exit 2 ;;
    esac
    mkdir -p "${enh_exp}"
    opts=()
    for phase in train valid; do
        if [[ ${phase} == train ]]; then split=tr; else split=cv; fi
        folder=${dumpdir}/raw/${split}_${mixture}
        opts+=(--${phase}_data_path_and_name_and_type "${folder}/wav.scp,speech_mix,sound")
        for speaker in 1 2; do
            opts+=(--${phase}_data_path_and_name_and_type "${folder}/spk${speaker}.scp,speech_ref${speaker},sound")
        done
        opts+=(--${phase}_data_path_and_name_and_type "${folder}/rir_ctf.scp,rir_ctf,npy")
        for field in speech_mix speech_ref1 speech_ref2; do
            opts+=(--${phase}_shape_file "${folder}/utt2num_samples")
        done
        opts+=(--${phase}_shape_file "${folder}/rir_ctf_shape")
    done
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${enh_exp}/train.log" \
        --log "${enh_exp}/train.log" --ngpu 1 --num_nodes 1 \
        --init_file_prefix "${enh_exp}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.enh_rir_ctf_train \
        --config "${enh_config}" --output_dir "${enh_exp}" --resume "${resume}" \
        --fold_length 200000 --fold_length 200000 --fold_length 200000 --fold_length 1000 \
        --batch_size 4 --valid_batch_size 1 --accum_grad 1 "${opts[@]}"
fi
for split in cv tt; do
    folder=${dumpdir}/raw/${split}_${mixture}
    out=${enh_exp}/inference_${split}
    if (( stage <= 7 && stop_stage >= 7 )); then
        "${python}" -m espnet2.bin.enh_rir_ctf_inference \
            --ngpu "${ngpu}" --fs 8000 --batch_size 1 --normalize_output_wav true \
            --train_config "${enh_exp}/config.yaml" --model_file "${enh_exp}/${inference_model}" \
            --data_path_and_name_and_type "${folder}/wav.scp,speech_mix,sound" \
            --data_path_and_name_and_type "${folder}/rir_ctf.scp,rir_ctf,npy" \
            --key_file "${folder}/wav.scp" --output_dir "${out}"
    fi
    if (( stage <= 8 && stop_stage >= 8 )); then
        "${python}" -m espnet2.bin.enh_scoring \
            --ref_scp "${folder}/spk1.scp" --ref_scp "${folder}/spk2.scp" \
            --inf_scp "${out}/spk1.scp" --inf_scp "${out}/spk2.scp" \
            --key_file "${folder}/wav.scp" --ref_channel 0 --output_dir "${out}/score"
    fi
done
