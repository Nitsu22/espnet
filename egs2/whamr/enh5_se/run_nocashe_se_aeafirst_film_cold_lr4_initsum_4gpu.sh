#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k

train_set=tr_mix_both_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_both_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_both_reverb_${min_or_max}_${sample_rate}"

init_param_src_path="../enh1/exp/enh_train_enh_tflocoformer_nocashe_2ch_4gpu/valid.loss.ave_5best.pth"
init_param_path="${init_param_src_path%.pth}.convsum_to_1ch.pth"

if [ ! -f "${init_param_path}" ]; then
    # Make sure torch is available (same as enh.sh does internally).
    . ./path.sh
    python3 ../tools/convsum_mc_to_sc_state_dict.py \
        --src "${init_param_src_path}" \
        --dst "${init_param_path}" \
        --module_key "separator.conv.0"
fi

init_param="${init_param_path}"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh_se_condition.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_nocashe_se_aeafirst_film_cold_lr4.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_nocashe_se_aeafusion_film_first_cold_lr4_initsum_4gpu \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --init_param "${init_param}" \
    --stage 6 \
    --stop_stage 8 \
    "$@"
