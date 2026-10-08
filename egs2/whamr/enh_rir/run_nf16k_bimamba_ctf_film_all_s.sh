#!/usr/bin/env bash
# Run from enh_rir, inside an appropriate allocation; do not select physical GPUs.
set -euo pipefail
stage=6
stop_stage=6
ngpu=1
conditioning=true
dumpdir=dump_nf_2spk_16k_min
enh_config=
enh_exp=
python=python3
inference_model=valid.loss.best.pth
. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

if [[ ${conditioning} == true ]]; then
    default_config=conf/tuning/train_enh_tflocoformer_s_nf16k_bimamba_ctf_film_all.yaml
    default_exp=exp/enh_train_tflocoformer_s_nf16k_bimamba_ctf_film_all_seed0
elif [[ ${conditioning} == false ]]; then
    default_config=../enh7_baseline/conf/tuning/train_enh_tflocoformer_s_nf_16k.yaml
    default_exp=exp/enh_train_tflocoformer_s_nf16k_baseline_seed0
else
    echo 'Use --conditioning true or false' >&2; exit 2
fi
enh_config=${enh_config:-${default_config}}
enh_exp=${enh_exp:-${default_exp}}
if (( stage < 5 || stop_stage > 8 || stage > stop_stage || ngpu < 0 )); then
    echo 'Use Stages 5-8 with a nonnegative GPU count' >&2; exit 2
fi
if (( stage <= 6 && stop_stage >= 6 && ngpu != 1 )); then
    echo 'Training matches enh7_baseline: one allocated GPU and physical batch four' >&2
    exit 2
fi
if [[ ! -L ${dumpdir} || ! -d ${dumpdir}/raw ]]; then
    echo "Missing linked dump: ${dumpdir}" >&2
    echo "Create: ln -s ../rir_2spk/dump_nf_2spk_16k_min ${dumpdir}" >&2
    exit 2
fi
mkdir -p "${enh_exp}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PWD}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.cache/matplotlib}"
if (( stage <= 5 && stop_stage >= 5 )); then
    "${python}" local/check_bimamba_conditioning_data.py \
        --dump "${dumpdir}" --output "${enh_exp}/data_check.json"
fi
if (( stage <= 6 && stop_stage >= 6 )); then
    # Header-derived mono sample lengths are exactly the baseline's three
    # shape series. No spectral means/variances are needed by either model.
    train=${dumpdir}/raw/tr_rir_2spk_nf_min_16k
    valid=${dumpdir}/raw/cv_rir_2spk_nf_min_16k
    opts=()
    for phase in train valid; do
        folder=${!phase}
        opts+=(--${phase}_data_path_and_name_and_type "${folder}/wav.scp,speech_mix,sound")
        for speaker in 1 2; do
            opts+=(--${phase}_data_path_and_name_and_type "${folder}/speech_direct${speaker}.scp,speech_ref${speaker},sound")
        done
        for field in speech_mix speech_ref1 speech_ref2; do
            opts+=(--${phase}_shape_file "${folder}/utt2num_samples")
        done
    done
    "${python}" -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${enh_exp}/train.log" \
        --log "${enh_exp}/train.log" --ngpu "${ngpu}" --num_nodes 1 \
        --init_file_prefix "${enh_exp}/.dist_init_" \
        --multiprocessing_distributed true -- \
        "${python}" -m espnet2.bin.enh_train \
        --config "${enh_config}" --output_dir "${enh_exp}" --resume true \
        --fold_length 400000 --fold_length 400000 --fold_length 400000 \
        --batch_size 4 --valid_batch_size 1 --accum_grad 1 "${opts[@]}"
fi
for split in cv tt; do
    data=${dumpdir}/raw/${split}_rir_2spk_nf_min_16k
    out=${enh_exp}/inference_${split}
    if (( stage <= 7 && stop_stage >= 7 )); then
        "${python}" -m espnet2.bin.enh_inference \
            --ngpu "${ngpu}" --fs 16000 --batch_size 1 \
            --train_config "${enh_exp}/config.yaml" \
            --model_file "${enh_exp}/${inference_model}" \
            --data_path_and_name_and_type "${data}/wav.scp,speech_mix,sound" \
            --key_file "${data}/wav.scp" --output_dir "${out}"
    fi
    if (( stage <= 8 && stop_stage >= 8 )); then
        "${python}" -m espnet2.bin.enh_scoring \
            --ref_scp "${data}/speech_direct1.scp" "${data}/speech_direct2.scp" \
            --inf_scp "${out}/spk1.scp" "${out}/spk2.scp" \
            --key_file "${data}/wav.scp" --ref_channel 0 \
            --output_dir "${out}/score"
    fi
done
