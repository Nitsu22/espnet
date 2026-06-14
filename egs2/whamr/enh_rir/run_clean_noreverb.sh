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
stage=3
stop_stage=4
dumpdir=dump_clean
expdir=exp_noreverb

train_set=tr_mix_clean_anechoic_${min_or_max}_${sample_rate}
valid_set=cv_mix_clean_anechoic_${min_or_max}_${sample_rate}
test_sets="tt_mix_clean_anechoic_${min_or_max}_${sample_rate}"

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_small.yaml \
    --expdir "${expdir}" \
    --enh_exp exp_noreverb/enh_train_enh_tflocoformer_small_clean_noreverb \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --dumpdir "${dumpdir}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --gpu_inference true \
    --inference_nj 1 \
    "$@"
