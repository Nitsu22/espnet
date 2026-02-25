#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k

train_set=tr_mix_clean_reverb_${min_or_max}_${sample_rate}
valid_set=cv_mix_clean_reverb_${min_or_max}_${sample_rate}
test_sets="tt_mix_clean_reverb_${min_or_max}_${sample_rate}"

enh_exp=exp/enh_train_enh_tflocoformer_nocashe_se_aeafusion_trainable_lr3_resnet_256_4gpu_small_clean_film2
inference_model=89epoch.pth
inference_tag=enhanced_89epoch_small_clean_film2

CUDA_VISIBLE_DEVICES=0 ./enh_se_condition.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_exp "${enh_exp}" \
    --inference_model "${inference_model}" \
    --inference_tag "${inference_tag}" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --audio_format wav \
    --dumpdir dump_clean \
    --gpu_inference true \
    --stage 7 \
    --stop_stage 8 \
    "$@"

