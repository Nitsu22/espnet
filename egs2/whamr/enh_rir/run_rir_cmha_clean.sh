#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k

train_set=tr_mix_clean_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_clean_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_clean_reverb_${min_or_max}_${sample_rate}"

prepare_data_link() {
    local dset=$1
    local src="${script_dir}/../enh1/data/${dset}"
    local dst="data/${dset}"

    if [ ! -e "${dst}" ]; then
        [ -d "${src}" ] || {
            echo "Error: ${src} is required. Run egs2/whamr/enh1 data prep first." >&2
            exit 2
        }
        mkdir -p data
        ln -s "${src}" "${dst}"
    fi

    for f in wav.scp spk1.scp spk2.scp; do
        [ -f "${dst}/${f}" ] || {
            echo "Error: ${dst}/${f} is required." >&2
            exit 2
        }
    done
}

prepare_data_link "${train_set}"
prepare_data_link "${valid_set}"
for dset in ${test_sets}; do
    prepare_data_link "${dset}"
done

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh_rir.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --rir_data_dir ../se_npz/data \
    --enh_config ./conf/tuning/train_enh_tflocoformer_small_rir_cmha.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_small_rir_cmha_clean \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --stage 3 \
    --stop_stage 8 \
    --gpu_inference true \
    --inference_nj 1 \
    "$@"
