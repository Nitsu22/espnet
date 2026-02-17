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

./enh_npz.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu 1 \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_se_mc_conformer_triplet_swap_samepair_noise_noseg.yaml \
    --enh_args "--preprocessor_conf speech_segment=null --preprocessor_conf contrastive_pool_npz_scp.tr=data/${train_set}/npz.scp --preprocessor_conf contrastive_pool_npz_scp.cv=data/${valid_set}/npz.scp" \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --expdir exp_swap_samepair \
    --enh_exp exp_swap_samepair/enh_train_se_mc_conformer_triplet_samepair_noise_noseg \
    --inference_model "valid.loss.best.pth" \
    --audio_format wav \
    --stage 6 \
    --stop_stage 6 \
    "$@"
