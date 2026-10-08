#!/usr/bin/env bash
set -euo pipefail
stage=6
stop_stage=6
ngpu=1
enh_exp=exp/enh_train_tflocoformer_s_nf_16k_1gpu_batch4
gpu_inference=true
inference_nj=1
. utils/parse_options.sh
[[ -f dump_nf16k/preparation.json ]] || { echo 'Run CPU preparation first'; exit 1; }
(( stage >= 6 && stop_stage <= 8 && stage <= stop_stage )) || exit 2
if (( stage == 6 )); then
    (( ngpu == 1 )) || exit 2
    for phase in train valid; do
        for field in speech_mix speech_ref1 speech_ref2; do
            [[ -s exp/enh_stats_16k/${phase}/${field}_shape ]] || exit 1
        done
    done
fi
bash ./enh.sh --stage "${stage}" --stop_stage "${stop_stage}" \
    --skip_data_prep true --skip_packing true --skip_upload_hf true \
    --fs 16k --ngpu "${ngpu}" --nj 4 --ref_num 2 \
    --train_set tr_mix_clean_reverb_min_16k \
    --valid_set cv_mix_clean_reverb_min_16k \
    --test_sets tt_mix_clean_reverb_min_16k \
    --dumpdir dump_nf16k --expdir exp --enh_exp "${enh_exp}" \
    --enh_config conf/tuning/train_enh_tflocoformer_s_nf_16k.yaml \
    --enh_speech_fold_length 4000 --use_dereverb_ref false --use_noise_ref false \
    --audio_format wav --inference_model valid.loss.best.pth \
    --gpu_inference "${gpu_inference}" --inference_nj "${inference_nj}" \
    --enh_args '--batch_size 4 --accum_grad 1 --valid_batch_size 1'
