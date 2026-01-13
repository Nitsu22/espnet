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
debug_dir=exp/enh_train_enh_tflocoformer_test4_4gpu_log/debug

# Debug controls for stall investigation
export ESPNET_LOG_ALL_RANKS=1
export ESPNET_DEBUG_STALL=1
export ESPNET_DEBUG_EVERY_N=1
export ESPNET_DEBUG_DIR="${debug_dir}"
export ESPNET_DATALOADER_TIMEOUT=300
export ESPNET_DATA_LOAD_WARN_SEC=5
export ESPNET_DEBUG_ITER_GAP_SEC=30

CUDA_VISIBLE_DEVICES=0,1,2,3 ./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 4 \
    --python ./local/python_log_wrapper.sh \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_tflocoformer_test4.yaml \
    --enh_exp exp/enh_train_enh_tflocoformer_test4_4gpu_log \
    --use_dereverb_ref false \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --stage 6 \
    --stop_stage 8 \
    "$@"
