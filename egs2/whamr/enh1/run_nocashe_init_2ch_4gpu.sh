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

# 2ch で学習したモデルで初期化し、separator の先頭 conv だけスクラッチで学習する。
# init_param 書式: <path>:<src_key>:<dst_key>:<exclude_keys>（コロンは3つで path:::exclude）
# exclude "separator.conv.0" で conv.0.weight / conv.0.bias をロード対象から外す。
init_param="exp/enh_train_enh_tflocoformer_nocashe_2ch_4gpu/valid.loss.best.pth:::separator.conv.0"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_nocashe.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_nocashe_2ch_init_4gpu \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --init_param "${init_param}" \
    --stage 6 \
    --stop_stage 8 \
    "$@"

